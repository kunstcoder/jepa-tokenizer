import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class JsonlAudioDataset(Dataset):
    """JSONL with {"path": ...}.

    If loading audio fails, it falls back to random noise for smoke reproducibility.
    """

    def __init__(self, jsonl_path: str, sample_rate: int = 24000, max_seconds: float = 2.0):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * max_seconds)
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item.get("path", "")
        wav = None
        if path and Path(path).exists():
            try:
                import torchaudio

                wav, sr = torchaudio.load(path)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            except Exception:
                wav = None
        if wav is None:
            wav = torch.randn(1, self.num_samples)
        if wav.shape[-1] < self.num_samples:
            wav = F.pad(wav, (0, self.num_samples - wav.shape[-1]))
        else:
            wav = wav[..., : self.num_samples]
        return wav


class DAAM(nn.Module):
    """Density-Adaptive attention-like gating using Gaussian mixture responses."""

    def __init__(self, channels: int, n_gaussians: int = 4):
        super().__init__()
        self.means = nn.Parameter(torch.randn(n_gaussians, channels))
        self.log_vars = nn.Parameter(torch.zeros(n_gaussians, channels))
        self.mix_logits = nn.Parameter(torch.zeros(n_gaussians))
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        b, t, c = x.shape
        x2 = x.unsqueeze(2)  # [B,T,1,C]
        mu = self.means.unsqueeze(0).unsqueeze(0)  # [1,1,G,C]
        var = self.log_vars.exp().unsqueeze(0).unsqueeze(0)
        log_prob = -0.5 * (((x2 - mu) ** 2) / (var + 1e-6) + self.log_vars.unsqueeze(0).unsqueeze(0)).sum(-1)
        mix = F.log_softmax(self.mix_logits, dim=0).view(1, 1, -1)
        weights = F.softmax(log_prob + mix, dim=-1)  # [B,T,G]
        confidence = weights.max(dim=-1, keepdim=True).values
        gated = x * confidence
        return self.proj(gated)


class SimpleEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )
        self.daam = DAAM(hidden)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.conv(wav)  # [B,C,T]
        z = z.transpose(1, 2)  # [B,T,C]
        z = self.daam(z)
        return z


class Predictor(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class FSQ(nn.Module):
    def __init__(self, levels: int = 8):
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x in R -> indices in [0, levels-1]
        clipped = torch.tanh(x)
        scaled = (clipped + 1.0) * 0.5 * (self.levels - 1)
        indices = torch.round(scaled).long().clamp(0, self.levels - 1)
        dequant = (indices.float() / (self.levels - 1)) * 2.0 - 1.0
        return dequant, indices


class Decoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(dim, 64, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=8, stride=4, padding=2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.transpose(1, 2))


@dataclass
class ModelBundle:
    encoder: SimpleEncoder
    predictor: Predictor
    fsq: FSQ
    decoder: Decoder


class JEPATokenizer(nn.Module):
    def __init__(self, hidden: int = 128, fsq_levels: int = 8):
        super().__init__()
        self.encoder = SimpleEncoder(hidden=hidden)
        self.predictor = Predictor(hidden)
        self.fsq = FSQ(fsq_levels)
        self.decoder = Decoder(hidden)

    def stage1_loss(self, wav: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
        z = self.encoder(wav)
        mask = torch.rand(z.shape[:2], device=z.device) < mask_ratio
        z_masked = z.clone()
        z_masked[mask] = 0.0
        pred = self.predictor(z_masked)
        return F.mse_loss(pred[mask], z[mask]) if mask.any() else F.mse_loss(pred, z)

    def encode_tokens(self, wav: torch.Tensor):
        z = self.encoder(wav)
        zq, idx = self.fsq(z)
        return zq, idx

    def reconstruct(self, wav: torch.Tensor):
        zq, idx = self.encode_tokens(wav)
        rec = self.decoder(zq)
        return rec, idx


def save_ckpt(path: str, model: nn.Module, step: int):
    torch.save({"step": step, "state_dict": model.state_dict()}, path)


def load_ckpt(path: str, model: nn.Module):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["state_dict"], strict=False)
    return data.get("step", 0)


def train_stage1(args):
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = JEPATokenizer(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=args.lr)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(99999):
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad()
            loss = model.stage1_loss(wav, mask_ratio=args.mask_ratio)
            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[stage1] step={step} loss={loss.item():.4f}")
            if step % args.save_every == 0:
                save_ckpt(str(Path(args.out_dir) / "stage1.pt"), model, step)
            if step >= args.max_steps:
                save_ckpt(str(Path(args.out_dir) / "stage1.pt"), model, step)
                return


def train_stage2(args):
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = JEPATokenizer(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)
    if args.stage1_ckpt:
        load_ckpt(args.stage1_ckpt, model)
    for p in model.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(list(model.fsq.parameters()) + list(model.decoder.parameters()), lr=args.lr)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(99999):
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad()
            rec, _ = model.reconstruct(wav)
            rec = rec[..., : wav.shape[-1]]
            loss = F.l1_loss(rec, wav)
            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[stage2] step={step} l1={loss.item():.4f}")
            if step % args.save_every == 0:
                save_ckpt(str(Path(args.out_dir) / "stage2.pt"), model, step)
            if step >= args.max_steps:
                save_ckpt(str(Path(args.out_dir) / "stage2.pt"), model, step)
                return


def run_infer(args):
    model = JEPATokenizer(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)
    load_ckpt(args.ckpt, model)
    model.eval()
    wav = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)[0].unsqueeze(0).to(args.device)
    with torch.no_grad():
        rec, idx = model.reconstruct(wav)
    rec = rec[..., : wav.shape[-1]]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_wav = Path(args.out_dir) / "recon.pt"
    out_idx = Path(args.out_dir) / "tokens.pt"
    torch.save(rec.cpu(), out_wav)
    torch.save(idx.cpu(), out_idx)
    print(f"saved: {out_wav}, {out_idx}; tokens_shape={tuple(idx.shape)}")


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--jsonl", required=True)
        sp.add_argument("--out_dir", required=True)
        sp.add_argument("--sample_rate", type=int, default=24000)
        sp.add_argument("--max_seconds", type=float, default=2.0)
        sp.add_argument("--batch_size", type=int, default=4)
        sp.add_argument("--hidden", type=int, default=128)
        sp.add_argument("--fsq_levels", type=int, default=8)
        sp.add_argument("--lr", type=float, default=1e-3)
        sp.add_argument("--max_steps", type=int, default=20)
        sp.add_argument("--save_every", type=int, default=10)
        sp.add_argument("--log_every", type=int, default=5)
        sp.add_argument("--device", default="cpu")

    s1 = sub.add_parser("train_stage1")
    common(s1)
    s1.add_argument("--mask_ratio", type=float, default=0.5)

    s2 = sub.add_parser("train_stage2")
    common(s2)
    s2.add_argument("--stage1_ckpt", default="")

    inf = sub.add_parser("infer")
    inf.add_argument("--jsonl", required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--out_dir", required=True)
    inf.add_argument("--sample_rate", type=int, default=24000)
    inf.add_argument("--max_seconds", type=float, default=2.0)
    inf.add_argument("--hidden", type=int, default=128)
    inf.add_argument("--fsq_levels", type=int, default=8)
    inf.add_argument("--device", default="cpu")
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train_stage1":
        train_stage1(args)
    elif args.cmd == "train_stage2":
        train_stage2(args)
    else:
        run_infer(args)


if __name__ == "__main__":
    main()
