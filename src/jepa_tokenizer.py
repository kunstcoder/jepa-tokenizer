import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ------------------------------
# Utilities (adapted from original repository naming)
# ------------------------------
def print_model_stats(model: nn.Module, stage_name: str = "Model") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print("\n" + "=" * 80)
    print(f"{stage_name} Architecture Summary")
    print("=" * 80)
    print(f"{'Component':<25} {'Parameters':<15} {'Trainable':<15}")
    print("-" * 80)

    for name, module in model.named_children():
        comp_params = sum(p.numel() for p in module.parameters())
        comp_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<25} {comp_params/1e6:>6.2f}M {comp_trainable/1e6:>6.2f}M")

    print("-" * 80)
    print(f"{'TOTAL':<25} {total_params/1e6:>6.2f}M {trainable_params/1e6:>6.2f}M")
    if frozen_params > 0:
        print(f"{'Frozen (lr=0)':<25} {frozen_params/1e6:>6.2f}M")
    print("=" * 80 + "\n")


@torch.no_grad()
def _fsq_dim_radices(dim: int, levels: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    assert dim % len(levels) == 0, f"dim={dim} must be divisible by len(levels)={len(levels)}"
    per = dim // len(levels)
    rad = []
    for level in levels:
        rad += [int(level)] * per
    return torch.tensor(rad, dtype=torch.long, device=device)


@torch.no_grad()
def fsq_pack_indices(indices: torch.Tensor, levels: List[int], group_size: int = 7) -> torch.Tensor:
    bsz, steps, dim = indices.shape
    device = indices.device
    rad = _fsq_dim_radices(dim, levels, device=device)

    groups = (dim + group_size - 1) // group_size
    pad = groups * group_size - dim
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(bsz, steps, pad, dtype=indices.dtype, device=device)], dim=2)
        rad = torch.cat([rad, torch.ones(pad, dtype=rad.dtype, device=device)], dim=0)

    toks = torch.zeros(bsz, steps, 0, dtype=torch.long, device=device)
    for g in range(groups):
        s, e = g * group_size, (g + 1) * group_size
        chunk = indices[:, :, s:e].long()
        rchunk = rad[s:e].long()
        tok = torch.zeros(bsz, steps, dtype=torch.long, device=device)
        for k in range(rchunk.numel() - 1, -1, -1):
            tok = chunk[:, :, k] + tok * rchunk[k]
        toks = torch.cat([toks, tok.unsqueeze(-1)], dim=-1)
    return toks


def fsq_token_stats_from_indices(
    indices: torch.Tensor,
    fsq_levels: List[int],
    code_dim: int,
    sample_rate: int,
    strides: List[int],
    group_size: int = 7,
) -> Dict[str, float]:
    hop = int(torch.tensor(strides).prod().item())
    fps = float(sample_rate) / float(hop)
    packed = fsq_pack_indices(indices, levels=fsq_levels, group_size=group_size)
    bsz, steps, groups = packed.shape
    return {
        "B": int(bsz),
        "T": int(steps),
        "G": int(groups),
        "fps": float(fps),
        "tokens_total": int(steps * groups),
        "tokens_per_sec": float(fps * groups),
        "group_size": int(group_size),
        "code_dim": int(code_dim),
        "hop": int(hop),
    }


def create_jepa_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.5,
    min_span: int = 4,
    max_span: int = 16,
    device: str = "cpu",
) -> torch.Tensor:
    """JEPA-style contiguous span masking for temporal sequences."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    target = max(1, int(seq_len * mask_ratio))

    for b in range(batch_size):
        covered = 0
        attempts = 0
        while covered < target and attempts < seq_len * 2:
            attempts += 1
            span = random.randint(min_span, max_span)
            span = min(span, seq_len)
            start = random.randint(0, max(0, seq_len - span))
            end = start + span
            before = mask[b].sum().item()
            mask[b, start:end] = True
            after = mask[b].sum().item()
            covered += int(after - before)
    return mask


# ------------------------------
# Data
# ------------------------------
class JsonlAudioDataset(Dataset):
    def __init__(self, jsonl_path: str, sample_rate: int = 24000, max_seconds: float = 2.0):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * max_seconds)
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
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


# ------------------------------
# Model blocks (names mirrored from original README/script)
# ------------------------------
class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, channels: int, n_gaussians: int = 4):
        super().__init__()
        self.means = nn.Parameter(torch.randn(n_gaussians, channels))
        self.log_vars = nn.Parameter(torch.zeros(n_gaussians, channels))
        self.mix_logits = nn.Parameter(torch.zeros(n_gaussians))
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.unsqueeze(2)
        mu = self.means.unsqueeze(0).unsqueeze(0)
        var = self.log_vars.exp().unsqueeze(0).unsqueeze(0)
        log_prob = -0.5 * (((x2 - mu) ** 2) / (var + 1e-6) + self.log_vars.unsqueeze(0).unsqueeze(0)).sum(-1)
        mix = F.log_softmax(self.mix_logits, dim=0).view(1, 1, -1)
        weights = F.softmax(log_prob + mix, dim=-1)
        confidence = weights.max(dim=-1, keepdim=True).values
        return self.proj(x * confidence)


class TinyConformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ff1 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.ff2 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 0.5 * self.ff1(self.norm(x))
        y = y + self.dwconv(y.transpose(1, 2)).transpose(1, 2)
        y = y + 0.5 * self.ff2(self.norm(y))
        return y


class JEPAEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 192):
        super().__init__()
        self.strides = [4, 4]
        self.frontend = nn.Sequential(
            nn.Conv1d(in_ch, 96, kernel_size=7, stride=self.strides[0], padding=3),
            nn.GELU(),
            nn.Conv1d(96, hidden, kernel_size=5, stride=self.strides[1], padding=2),
            nn.GELU(),
        )
        self.gaatn = GaussianAdaptiveAttention(hidden)
        self.conformer = TinyConformerBlock(hidden)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.frontend(wav).transpose(1, 2)
        z = self.gaatn(z)
        z = self.conformer(z)
        return z


class JEPAPredictor(nn.Module):
    def __init__(self, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQ(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.level = int(levels[0])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        clipped = torch.tanh(x)
        scaled = (clipped + 1.0) * 0.5 * (self.level - 1)
        indices = torch.round(scaled).long().clamp(0, self.level - 1)
        dequant = (indices.float() / (self.level - 1)) * 2.0 - 1.0
        return dequant, indices


class SimpleHiFiGenerator(nn.Module):
    def __init__(self, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(dim, 96, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(96, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.transpose(1, 2))


class JEPAFSQVAE(nn.Module):
    def __init__(self, hidden: int = 192, fsq_levels: Optional[List[int]] = None):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [8, 8, 8, 8]
        self.encoder = JEPAEncoder(hidden=hidden)
        self.predictor = JEPAPredictor(hidden)
        self.fsq = FSQ(fsq_levels)
        self.decoder = SimpleHiFiGenerator(hidden)
        self.fsq_levels = fsq_levels

    def stage1_loss(self, wav: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
        z = self.encoder(wav)
        mask = create_jepa_mask(z.shape[0], z.shape[1], mask_ratio=mask_ratio, device=str(z.device))
        z_masked = z.clone()
        z_masked[mask] = 0.0
        pred = self.predictor(z_masked)
        return F.mse_loss(pred[mask], z[mask]) if mask.any() else F.mse_loss(pred, z)

    def encode_tokens(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(wav)
        zq, indices = self.fsq(z)
        return zq, indices

    def reconstruct(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zq, indices = self.encode_tokens(wav)
        rec = self.decoder(zq)
        return rec, indices


# ------------------------------
# Train / infer
# ------------------------------
def save_ckpt(path: str, model: nn.Module, step: int) -> None:
    torch.save({"step": step, "state_dict": model.state_dict()}, path)


def load_ckpt(path: str, model: nn.Module) -> int:
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["state_dict"], strict=False)
    return int(data.get("step", 0))


def train_jepa(args: argparse.Namespace) -> None:
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = JEPAFSQVAE(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)
    print_model_stats(model, "Stage 1 (JEPA)")

    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=args.lr)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for _ in range(999999):
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad()
            loss = model.stage1_loss(wav, mask_ratio=args.mask_ratio)
            loss.backward()
            opt.step()
            step += 1

            if step % args.log_every == 0:
                print(f"[train_jepa] step={step} loss={loss.item():.4f}")
            if step % args.save_every_steps == 0:
                save_ckpt(str(Path(args.out_dir) / "stage1_jepa.pt"), model, step)
            if step >= args.max_steps:
                save_ckpt(str(Path(args.out_dir) / "stage1_jepa.pt"), model, step)
                return


def train_decoder(args: argparse.Namespace) -> None:
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = JEPAFSQVAE(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)

    if args.stage1_ckpt:
        load_ckpt(args.stage1_ckpt, model)

    for p in model.encoder.parameters():
        p.requires_grad = False

    print_model_stats(model, "Stage 2 (Decoder)")
    opt = torch.optim.AdamW(list(model.fsq.parameters()) + list(model.decoder.parameters()), lr=args.lr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    for _ in range(999999):
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad()
            rec, idx = model.reconstruct(wav)
            rec = rec[..., : wav.shape[-1]]

            l1 = F.l1_loss(rec, wav)
            spectral = F.mse_loss(torch.fft.rfft(rec, dim=-1).abs(), torch.fft.rfft(wav, dim=-1).abs())
            loss = l1 + args.spectral_weight * spectral
            loss.backward()
            opt.step()

            step += 1
            if step % args.log_every == 0:
                stats = fsq_token_stats_from_indices(
                    idx,
                    fsq_levels=args.fsq_levels,
                    code_dim=idx.shape[-1],
                    sample_rate=args.sample_rate,
                    strides=model.encoder.strides,
                )
                print(
                    f"[train_decoder] step={step} loss={loss.item():.4f} l1={l1.item():.4f} "
                    f"tokens/sec={stats['tokens_per_sec']:.2f}"
                )
            if step % args.save_every_steps == 0:
                save_ckpt(str(Path(args.out_dir) / "stage2_decoder.pt"), model, step)
            if step >= args.max_steps:
                save_ckpt(str(Path(args.out_dir) / "stage2_decoder.pt"), model, step)
                return


def run_infer(args: argparse.Namespace) -> None:
    model = JEPAFSQVAE(hidden=args.hidden, fsq_levels=args.fsq_levels).to(args.device)
    load_ckpt(args.ckpt, model)
    model.eval()

    wav = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)[0].unsqueeze(0).to(args.device)
    with torch.no_grad():
        rec, idx = model.reconstruct(wav)
    rec = rec[..., : wav.shape[-1]]
    packed = fsq_pack_indices(idx, args.fsq_levels)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(rec.cpu(), Path(args.out_dir) / "recon.pt")
    torch.save(idx.cpu(), Path(args.out_dir) / "indices.pt")
    torch.save(packed.cpu(), Path(args.out_dir) / "packed_tokens.pt")
    print(f"saved recon/indices/packed_tokens in {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Density-Adaptive JEPA tokenizer (original-code-oriented baseline)")
    p.add_argument("--stage", choices=["train_jepa", "train_decoder", "infer"], required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--max_seconds", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--fsq_levels", type=int, nargs="+", default=[8, 8, 8, 8])
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--save_every_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cpu")

    # Stage 1
    p.add_argument("--mask_ratio", type=float, default=0.5)

    # Stage 2
    p.add_argument("--stage1_ckpt", default="")
    p.add_argument("--spectral_weight", type=float, default=2.0)

    # Infer
    p.add_argument("--ckpt", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage == "train_jepa":
        train_jepa(args)
    elif args.stage == "train_decoder":
        train_decoder(args)
    else:
        if not args.ckpt:
            raise ValueError("--ckpt is required for --stage infer")
        run_infer(args)


if __name__ == "__main__":
    main()
