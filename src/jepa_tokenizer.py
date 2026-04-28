import argparse
import copy
import json
import math
import os
import random
import time
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
    print("\n" + "=" * 96)
    print(f"{stage_name} Architecture Summary")
    print("=" * 96)
    print(f"{'Component':<32} {'Parameters':<16} {'Trainable':<16}")
    print("-" * 96)

    for name, module in model.named_children():
        comp_params = sum(p.numel() for p in module.parameters())
        comp_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<32} {comp_params/1e6:>7.3f}M {comp_trainable/1e6:>7.3f}M")

    print("-" * 96)
    print(f"{'TOTAL':<32} {total_params/1e6:>7.3f}M {trainable_params/1e6:>7.3f}M")
    if frozen_params > 0:
        print(f"{'Frozen (lr=0)':<32} {frozen_params/1e6:>7.3f}M")
    print("=" * 96 + "\n")


@torch.no_grad()
def _fsq_dim_radices(dim: int, levels: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    assert len(levels) > 0, "levels must not be empty"
    repeat = math.ceil(dim / len(levels))
    radices = (levels * repeat)[:dim]
    return torch.tensor(radices, dtype=torch.long, device=device)


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

    toks = torch.zeros(bsz, steps, groups, dtype=torch.long, device=device)
    for g in range(groups):
        s, e = g * group_size, (g + 1) * group_size
        chunk = indices[:, :, s:e].long()
        rchunk = rad[s:e].long()
        tok = torch.zeros(bsz, steps, dtype=torch.long, device=device)
        for k in range(rchunk.numel() - 1, -1, -1):
            tok = chunk[:, :, k] + tok * rchunk[k]
        toks[:, :, g] = tok
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
    max_span: int = 24,
    device: str = "cpu",
) -> torch.Tensor:
    """JEPA-style contiguous span masking for temporal sequences."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    target = max(1, int(seq_len * mask_ratio))

    for b in range(batch_size):
        covered = 0
        attempts = 0
        while covered < target and attempts < seq_len * 4:
            attempts += 1
            span = min(random.randint(min_span, max_span), seq_len)
            start = random.randint(0, max(0, seq_len - span))
            before = mask[b].sum().item()
            mask[b, start : start + span] = True
            covered += int(mask[b].sum().item() - before)
    return mask


# ------------------------------
# Data
# ------------------------------
class JsonlAudioDataset(Dataset):
    def __init__(self, jsonl_path: str, sample_rate: int = 24000, max_seconds: float = 2.0):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * max_seconds)
        self.items: List[Dict[str, str]] = []
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
# Model blocks (original naming-aligned but stronger baseline)
# ------------------------------
class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, channels: int, n_gaussians: int = 8):
        super().__init__()
        self.means = nn.Parameter(torch.randn(n_gaussians, channels) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(n_gaussians, channels))
        self.mix_logits = nn.Parameter(torch.zeros(n_gaussians))
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x2 = x.unsqueeze(2)
        mu = self.means.unsqueeze(0).unsqueeze(0)
        var = self.log_vars.exp().unsqueeze(0).unsqueeze(0)
        mahal = ((x2 - mu) ** 2) / (var + 1e-6)
        log_prob = -0.5 * (mahal + self.log_vars.unsqueeze(0).unsqueeze(0)).sum(-1)
        mix = F.log_softmax(self.mix_logits, dim=0).view(1, 1, -1)
        weights = F.softmax(log_prob + mix, dim=-1)
        conf = weights.max(dim=-1, keepdim=True).values
        return self.out_proj(x * conf)


class ConvSubsampler(nn.Module):
    def __init__(self, in_ch: int, hidden: int, strides: List[int]):
        super().__init__()
        self.strides = strides
        blocks = []
        ch = in_ch
        cur = hidden // 2
        for i, s in enumerate(strides):
            out_ch = hidden if i == len(strides) - 1 else min(hidden, cur * 2)
            blocks.extend([
                nn.Conv1d(ch, out_ch, kernel_size=7 if i == 0 else 5, stride=s, padding=3 if i == 0 else 2),
                nn.GELU(),
            ])
            ch = out_ch
            cur = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.mha_ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.conv_ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.mha_ln(x)
        attn_out, _ = self.mha(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out

        y = self.conv_ln(x).transpose(1, 2)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y).transpose(1, 2)
        x = x + y

        x = x + 0.5 * self.ffn2(x)
        return self.out_ln(x)


class JEPAEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        hidden: int = 256,
        strides: Optional[List[int]] = None,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if strides is None:
            strides = [2, 2, 2, 2]
        self.strides = strides
        self.frontend = ConvSubsampler(in_ch=in_ch, hidden=hidden, strides=strides)
        self.gaatn = GaussianAdaptiveAttention(hidden)
        self.blocks = nn.ModuleList([ConformerBlock(hidden, n_heads=n_heads, dropout=dropout) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(hidden)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.frontend(wav)
        z = self.gaatn(z)
        for blk in self.blocks:
            z = blk(z)
        return self.final_ln(z)


class JEPAPredictor(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([ConformerBlock(dim, n_heads=n_heads, conv_kernel=9, dropout=dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x.clone()
        h[mask] = self.mask_token.expand_as(h)[mask]
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class FSQ(nn.Module):
    def __init__(self, levels: List[int], code_dim: int):
        super().__init__()
        self.levels = levels
        self.code_dim = code_dim

    def _levels_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        return _fsq_dim_radices(self.code_dim, self.levels, x.device).float().view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        levels = self._levels_per_dim(x)
        clipped = torch.tanh(x)
        scaled = (clipped + 1.0) * 0.5 * (levels - 1.0)
        indices = torch.round(scaled).long().clamp_min(0)
        indices = torch.minimum(indices, (levels.long() - 1))
        dequant = (indices.float() / (levels - 1.0).clamp_min(1.0)) * 2.0 - 1.0
        # straight-through estimator
        zq = x + (dequant - x).detach()
        return zq, indices


class SimpleHiFiGenerator(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(dim, dim // 2, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(dim // 2, dim // 4, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(dim // 4, dim // 8, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(dim // 8, 32, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.transpose(1, 2))


class JEPAFSQVAE(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        fsq_levels: Optional[List[int]] = None,
        enc_depth: int = 4,
        pred_depth: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [8, 8, 8, 8]

        self.encoder = JEPAEncoder(hidden=hidden, depth=enc_depth, n_heads=n_heads, dropout=dropout)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(hidden, depth=pred_depth, n_heads=n_heads, dropout=dropout)
        self.fsq = FSQ(fsq_levels, code_dim=hidden)
        self.decoder = SimpleHiFiGenerator(hidden)
        self.fsq_levels = fsq_levels

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.995) -> None:
        for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def stage1_loss(self, wav: torch.Tensor, mask_ratio: float = 0.5) -> Tuple[torch.Tensor, Dict[str, float]]:
        online = self.encoder(wav)
        with torch.no_grad():
            target = self.target_encoder(wav)

        mask = create_jepa_mask(online.shape[0], online.shape[1], mask_ratio=mask_ratio, device=str(online.device))
        pred = self.predictor(online, mask)

        if mask.any():
            recon = F.smooth_l1_loss(pred[mask], target[mask])
            align = 1.0 - F.cosine_similarity(pred[mask], target[mask], dim=-1).mean()
        else:
            recon = F.smooth_l1_loss(pred, target)
            align = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        loss = recon + 0.2 * align
        return loss, {
            "recon": float(recon.detach().item()),
            "align": float(align.detach().item()),
            "mask_ratio_real": float(mask.float().mean().item()),
        }

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
def save_ckpt(path: str, model: nn.Module, step: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "step": step,
            "state_dict": model.state_dict(),
            "config": {
                "hidden": args.hidden,
                "enc_depth": args.enc_depth,
                "pred_depth": args.pred_depth,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
                "fsq_levels": args.fsq_levels,
            },
        },
        path,
    )


def ckpt_path_with_step(out_dir: str, prefix: str, step: int) -> str:
    return str(Path(out_dir) / f"{prefix}_step{step}.pt")


def load_ckpt(path: str, model: nn.Module) -> int:
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["state_dict"], strict=False)
    return int(data.get("step", 0))


def train_jepa(args: argparse.Namespace) -> None:
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = JEPAFSQVAE(
        hidden=args.hidden,
        fsq_levels=args.fsq_levels,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(args.device)
    print_model_stats(model, "Stage 1 (JEPA)")

    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=args.lr)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss, aux = model.stage1_loss(wav, mask_ratio=args.mask_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.predictor.parameters()), args.grad_clip)
            opt.step()
            model.update_target_encoder(momentum=args.ema_momentum)
            step += 1

            if step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - t0)
                print(
                    f"[train_jepa] step={step} loss={loss.item():.4f} recon={aux['recon']:.4f} "
                    f"align={aux['align']:.4f} mask={aux['mask_ratio_real']:.3f} steps/s={step/elapsed:.2f}"
                )
            if step % args.save_every_steps == 0 or step >= args.max_steps:
                save_ckpt(ckpt_path_with_step(args.out_dir, "stage1_jepa", step), model, step, args)
            if step >= args.max_steps:
                break


def train_decoder(args: argparse.Namespace) -> None:
    ds = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = JEPAFSQVAE(
        hidden=args.hidden,
        fsq_levels=args.fsq_levels,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(args.device)

    if args.stage1_ckpt:
        load_ckpt(args.stage1_ckpt, model)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False

    print_model_stats(model, "Stage 2 (Decoder)")
    opt = torch.optim.AdamW(list(model.fsq.parameters()) + list(model.decoder.parameters()), lr=args.lr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for wav in dl:
            wav = wav.to(args.device)
            opt.zero_grad(set_to_none=True)
            rec, idx = model.reconstruct(wav)
            rec = rec[..., : wav.shape[-1]]

            l1 = F.l1_loss(rec, wav)
            spec_rec = torch.log1p(torch.fft.rfft(rec, dim=-1).abs())
            spec_wav = torch.log1p(torch.fft.rfft(wav, dim=-1).abs())
            spectral = F.mse_loss(spec_rec, spec_wav)
            loss = l1 + args.spectral_weight * spectral
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.fsq.parameters()) + list(model.decoder.parameters()), args.grad_clip)
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
                elapsed = max(1e-6, time.time() - t0)
                print(
                    f"[train_decoder] step={step} loss={loss.item():.4f} l1={l1.item():.4f} "
                    f"spec={spectral.item():.4f} tok/s={stats['tokens_per_sec']:.2f} steps/s={step/elapsed:.2f}"
                )
            if step % args.save_every_steps == 0 or step >= args.max_steps:
                save_ckpt(ckpt_path_with_step(args.out_dir, "stage2_decoder", step), model, step, args)
            if step >= args.max_steps:
                break


def run_infer(args: argparse.Namespace) -> None:
    model = JEPAFSQVAE(
        hidden=args.hidden,
        fsq_levels=args.fsq_levels,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(args.device)
    load_ckpt(args.ckpt, model)
    model.eval()

    wav = JsonlAudioDataset(args.jsonl, args.sample_rate, args.max_seconds)[0].unsqueeze(0).to(args.device)
    with torch.no_grad():
        rec, idx = model.reconstruct(wav)
    rec = rec[..., : wav.shape[-1]]
    packed = fsq_pack_indices(idx, args.fsq_levels, group_size=args.pack_group_size)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(rec.cpu(), Path(args.out_dir) / "recon.pt")
    torch.save(idx.cpu(), Path(args.out_dir) / "indices.pt")
    torch.save(packed.cpu(), Path(args.out_dir) / "packed_tokens.pt")
    print(f"saved recon/indices/packed_tokens in {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Density-Adaptive JEPA tokenizer (original-code-oriented rebuild)")
    p.add_argument("--stage", choices=["train_jepa", "train_decoder", "infer"], required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--max_seconds", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--enc_depth", type=int, default=4)
    p.add_argument("--pred_depth", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fsq_levels", type=int, nargs="+", default=[8, 8, 8, 8])
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--save_every_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")

    # Stage 1
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--ema_momentum", type=float, default=0.995)

    # Stage 2
    p.add_argument("--stage1_ckpt", default="")
    p.add_argument("--spectral_weight", type=float, default=2.0)

    # Infer
    p.add_argument("--ckpt", default="")
    p.add_argument("--pack_group_size", type=int, default=7)
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
