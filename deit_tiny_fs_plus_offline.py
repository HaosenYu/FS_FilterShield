#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DeiT-tiny + CIFAR-10
# Backdoor Attacks (BadNets/WaNet/Blend/SIG/TrojViT/BadViT) +
# FilterShield (FS) universal defense +
# Evaluation + Trigger visualization
#
# FS (universal version) high-level:
#   - Freeze victim model, insert a tiny learnable linear "FilterShield" P on tokens.
#   - Train P on a small clean subset only.
#   - During FS training:
#       * Randomly inject lightweight "proxy triggers" (visible patch, invisible patch, local texture blob...)
#         so the model sees trigger-like artifacts without needing the real attacker trigger.
#       * Run strong PGD from those proxy-triggered samples to expose vulnerable directions.
#       * Optimize P with:
#             CE_clean            (keep clean accuracy)
#           + lambda1 * CE_adv    (keep semantics under stress)
#           + lambda_consistency * KL(logits_adv || logits_clean)    (stay consistent; don't jump class)
#           + lambda_entropy * entropy_loss(logits_adv)              (avoid overconfident single-class collapse)
#           + lambda3 * ||P - I||^2                                  (stay near identity so we don't wreck clean)
#
#   => No need to know which class is the attack target.
#   => Aims to generalize to many trigger styles (BadNets corners, TrojViT blobs, BadViT geometric patterns,
#      invisible patch noise, smooth warps like WaNet, etc.).
#
# Example victim training:
#   (BadViT visible)
#   python deit_tiny_fs_plus.py --task train_victim --data ./data --epochs 100 \
#     --poison_rate 0.1 --target 0 --pretrained \
#     --attack badvit --badvit_mode visible --badvit_shape cross --badvit_alpha 0.8 \
#     --save victim_badvit_visible.pth
#
#   (TrojViT style backdoor)
#   python deit_tiny_fs_plus.py --task train_victim --data ./data --epochs 100 \
#     --poison_rate 0.1 --target 0 --pretrained \
#     --attack trojvit --trojvit_center_row 12 --trojvit_center_col 12 \
#     --trojvit_radius_patches 1 --trojvit_alpha 0.9 \
#     --save victim_trojvit.pth
#
# Train FilterShield (universal mode):
#   python deit_tiny_fs_plus.py --task train_fs --data ./data \
#     --victim_ckpt victim_trojvit.pth --save_fs fs_trojvit_univ.pth \
#     --clean_fraction 0.10 --fs_epochs 15 --fs_after_block 10 --fs_split_cls \
#     --use_ln_after_fs \
#     --eps 0.047 --alpha 0.007 --steps 20 \
#     --lambda1 1.0 --lambda_consistency 1.0 --lambda_entropy 0.5 --lambda3 1e-4 \
#     --proxy_triggers_prob 0.5
#
# Eval + visualize:
#   python deit_tiny_fs_plus.py --task eval --data ./data \
#     --victim_ckpt victim_trojvit.pth --fs_ckpt fs_trojvit_univ.pth \
#     --attack trojvit --target 0 \
#     --viz_triggers 12 --viz_outdir ./viz_trojvit_univ
#
import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image, make_grid

import timm

from torchvision.utils import save_image
# import matplotlib.pyplot as plt
# import numpy as np

# -----------------------------
# Offline support
# -----------------------------
def _set_offline_env():
    # Hugging Face Hub and Transformers honor these environment variables.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _assert_cifar10_present(root: str):
    # torchvision CIFAR10 expects these files under root/cifar-10-batches-py/
    expected_dir = os.path.join(root, "cifar-10-batches-py")
    required = [
        "batches.meta",
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    ]
    if not os.path.isdir(expected_dir):
        raise FileNotFoundError(
            f"CIFAR-10 not found at {expected_dir}. "
            f"On a machine with internet, run: "
            f"python -c \"import torchvision; torchvision.datasets.CIFAR10(root='{root}', train=True, download=True); torchvision.datasets.CIFAR10(root='{root}', train=False, download=True)\" "
            f"then copy the whole '{expected_dir}' folder into the data center."
        )
    missing = [f for f in required if not os.path.isfile(os.path.join(expected_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"CIFAR-10 folder exists but is missing files: {missing}. "
            f"Please re-download on a machine with internet and copy the full folder."
        )

def _load_local_checkpoint(model: nn.Module, ckpt_path: str):
    # Prefer safe weight-only loading when supported (PyTorch 2.1+)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    # Remove possible prefixes and drop incompatible tensors (e.g., classifier head)
    model_state = model.state_dict()
    new_state = {}
    skipped = []

    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk in model_state and hasattr(v, "shape") and model_state[nk].shape != v.shape:
            skipped.append((nk, tuple(v.shape), tuple(model_state[nk].shape)))
            continue
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    if skipped:
        print(f"[offline] skipped {len(skipped)} keys due to shape mismatch (likely classifier head). Examples:")
        for nk, src_shape, dst_shape in skipped[:6]:
            print(f"  {nk}: ckpt{src_shape} -> model{dst_shape}")

    if missing:
        print(f"[offline] missing keys when loading ckpt: {len(missing)}")
    if unexpected:
        print(f"[offline] unexpected keys when loading ckpt: {len(unexpected)}")
    return model


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm(x: torch.Tensor, mean, std):
    # x: (B,C,H,W) normalized
    mean = torch.tensor(mean, device=x.device)[None, :, None, None]
    std = torch.tensor(std, device=x.device)[None, :, None, None]
    return (x * std) + mean


def renorm(x: torch.Tensor, mean, std):
    # x: (B,C,H,W) pixel space
    mean = torch.tensor(mean, device=x.device)[None, :, None, None]
    std = torch.tensor(std, device=x.device)[None, :, None, None]
    return (x - mean) / std


def random_erasing_pixelspace(x_pix: torch.Tensor, p: float = 0.5,
                              scale=(0.02, 0.15), ratio=(0.3, 3.3)) -> torch.Tensor:
    """Random erasing in pixel space [0,1]."""
    if torch.rand(1, device=x_pix.device).item() > p:
        return x_pix
    B, C, H, W = x_pix.shape
    out = x_pix.clone()
    area = H * W
    for b in range(B):
        target_area = area * (scale[0] + (scale[1] - scale[0]) * torch.rand(1, device=x_pix.device).item())
        aspect = ratio[0] + (ratio[1] - ratio[0]) * torch.rand(1, device=x_pix.device).item()
        h = int(round((target_area * aspect) ** 0.5))
        w = int(round((target_area / aspect) ** 0.5))
        if h <= 0 or w <= 0 or h >= H or w >= W:
            continue
        top = torch.randint(0, H - h, (1,), device=x_pix.device).item()
        left = torch.randint(0, W - w, (1,), device=x_pix.device).item()
        out[b, :, top:top + h, left:left + w] = torch.rand((C, h, w), device=x_pix.device)
    return out


def common_degrade_pixelspace(x_pix: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """Light, shape-agnostic degradations in pixel space [0,1]."""
    if torch.rand(1, device=x_pix.device).item() > prob:
        return x_pix
    out = x_pix
    # Mild blur sometimes
    if torch.rand(1, device=x_pix.device).item() < 0.5:
        k = 3 if torch.rand(1, device=x_pix.device).item() < 0.5 else 5
        out = TF.gaussian_blur(out, kernel_size=[k, k], sigma=[0.1, 1.2])
    # Random erasing
    out = random_erasing_pixelspace(out, p=0.7)
    return out.clamp(0.0, 1.0)


# -----------------------------
# Proxy trigger classes (for FS training only)
# These are DIFFERENT from the evaluation attacks to avoid overfitting.
# Based on: FTrojan, ISSBA-inspired Perlin noise, random affine, color shift, grid dropout.
# -----------------------------

class FTrojanProxy:
    """FTrojan-style frequency-domain trigger injection via torch.fft (no cv2 dependency).
    Adds fixed magnitude at specific frequency positions in each image block.
    Ref: Wang et al., "An Invisible Black-box Backdoor Attack through Frequency Domain"
    """
    def __init__(self, magnitude: float = 0.04,
                 freq_positions: tuple = ((2, 2), (4, 1), (1, 4)),
                 window: int = 8):
        self.mag = magnitude
        self.freq_pos = freq_positions
        self.ws = window

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        out = img.clone()
        for w0 in range(0, H, self.ws):
            for h0 in range(0, W, self.ws):
                block = out[:, :, w0:w0 + self.ws, h0:h0 + self.ws]
                fft_block = torch.fft.fft2(block)
                for (r, c) in self.freq_pos:
                    if r < self.ws and c < self.ws:
                        fft_block[:, :, r, c] = fft_block[:, :, r, c] + self.mag
                out[:, :, w0:w0 + self.ws, h0:h0 + self.ws] = torch.fft.ifft2(fft_block).real
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class PerlinNoiseProxy:
    """Low-frequency smooth noise texture overlay (simplified Perlin-style).
    Inspired by ISSBA sample-specific triggers, simplified to fixed noise pattern.
    """
    def __init__(self, alpha: float = 0.15, grid_size: int = 8, seed: int = 42):
        self.alpha = alpha
        self.grid_size = grid_size
        self.seed = seed
        self.cached = {}

    def _make_noise(self, H: int, W: int, dev: torch.device) -> torch.Tensor:
        key = (H, W, dev.type)
        if key in self.cached:
            return self.cached[key]
        rng = torch.Generator(device=dev)
        rng.manual_seed(self.seed)
        coarse = torch.randn(1, 1, self.grid_size, self.grid_size, generator=rng, device=dev)
        noise = F.interpolate(coarse, size=(H, W), mode='bicubic', align_corners=True)[0, 0]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        self.cached[key] = noise
        return noise

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        noise = self._make_noise(H, W, img.device).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        out = (1 - self.alpha) * img + self.alpha * noise
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class RandomAffineProxy:
    """Random affine warp (rotation + scale + translate).
    Different mechanism from WaNet (which uses fixed elastic displacement field).
    """
    def __init__(self, max_rotate: float = 8.0, max_scale: float = 0.08,
                 max_translate: float = 0.06):
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.max_translate = max_translate

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        dev = img.device
        angle = (torch.rand(B, device=dev) * 2 - 1) * self.max_rotate
        scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * self.max_scale
        tx = (torch.rand(B, device=dev) * 2 - 1) * self.max_translate
        ty = (torch.rand(B, device=dev) * 2 - 1) * self.max_translate
        cos_a = torch.cos(angle * torch.pi / 180)
        sin_a = torch.sin(angle * torch.pi / 180)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = -sin_a * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, img.shape, align_corners=True)
        out = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection',
                            align_corners=True)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class ColorShiftProxy:
    """Shift each RGB channel independently by a fixed delta.
    No evaluation attack uses channel-level color perturbation.
    """
    def __init__(self, max_delta: float = 0.08, seed: int = 42):
        self.max_delta = max_delta
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        rng = torch.Generator(device=img.device)
        rng.manual_seed(self.seed)
        delta = (torch.rand(1, C, 1, 1, generator=rng, device=img.device) * 2 - 1) * self.max_delta
        out = img + delta
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class GridDropoutProxy:
    """Structured grid-based dropout/masking (GridMask-style).
    Different from any evaluation attack's additive perturbation mode.
    """
    def __init__(self, grid_d: int = 16, mask_ratio: float = 0.3,
                 fill: float = 0.0, seed: int = 42):
        self.grid_d = grid_d
        self.mask_ratio = mask_ratio
        self.fill = fill
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        dev = img.device
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)
        gh, gw = H // self.grid_d, W // self.grid_d
        mask_grid = (torch.rand(gh, gw, generator=rng) > self.mask_ratio).float()
        mask = mask_grid.repeat_interleave(self.grid_d, 0).repeat_interleave(self.grid_d, 1)
        mask = mask[:H, :W].unsqueeze(0).unsqueeze(0).to(dev)
        out = img * mask + self.fill * (1 - mask)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class RandomPatchProxy:
    """Random-location solid color patch — mimics strong local triggers like BadNets
    but at random positions/sizes/colors (not bottom-right white square).
    """
    def __init__(self, min_size: int = 8, max_size: int = 24, seed: int = 42):
        self.min_size = min_size
        self.max_size = max_size
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        out = img.clone()
        for b in range(B):
            s = random.randint(self.min_size, self.max_size)
            y0 = random.randint(0, H - s)
            x0 = random.randint(0, W - s)
            color = torch.rand(C, 1, 1, device=img.device)
            out[b, :, y0:y0 + s, x0:x0 + s] = color
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class PatchTokenProxy:
    """Precisely targets 1~few ViT patch-sized (16x16) regions with high-alpha pattern replacement.
    Designed to mimic TrojViT-style attacks: tiny region, near-complete overwrite.

    Key difference from TrojViT: uses random patch positions, random pattern types
    (horizontal stripes, gradient, random noise — NOT checker), random alpha per call.
    """
    def __init__(self, patch_size: int = 16, num_patches: int = 3,
                 alpha_range: tuple = (0.7, 0.95), seed: int = 42):
        self.ps = patch_size
        self.num_patches = num_patches
        self.alpha_lo, self.alpha_hi = alpha_range
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        dev = img.device
        gh, gw = H // self.ps, W // self.ps
        out = img.clone()

        for b in range(B):
            alpha = random.uniform(self.alpha_lo, self.alpha_hi)
            # Pick random patch positions
            chosen = random.sample(range(gh * gw), min(self.num_patches, gh * gw))
            # Random pattern type per sample
            ptype = random.choice(['hstripes', 'gradient', 'noise'])
            for idx in chosen:
                pr, pc = divmod(idx, gw)
                y0, x0 = pr * self.ps, pc * self.ps
                if ptype == 'hstripes':
                    yy = torch.arange(self.ps, device=dev).float()
                    pat = ((yy // 4) % 2).unsqueeze(1).expand(self.ps, self.ps)
                    pat = pat.unsqueeze(0).expand(C, -1, -1)
                elif ptype == 'gradient':
                    pat = torch.linspace(0, 1, self.ps, device=dev).unsqueeze(0).expand(self.ps, self.ps)
                    pat = pat.unsqueeze(0).expand(C, -1, -1)
                else:  # noise
                    pat = torch.rand(C, self.ps, self.ps, device=dev)
                out[b, :, y0:y0 + self.ps, x0:x0 + self.ps] = (
                    (1 - alpha) * out[b, :, y0:y0 + self.ps, x0:x0 + self.ps] + alpha * pat)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class LocalPatternProxy:
    """Local region filled with a high-contrast pattern (stripes/checker) at random position.
    Mimics TrojViT/BadViT-style localized pattern blending but at different locations/patterns.
    """
    def __init__(self, region_frac: float = 0.15, alpha: float = 0.8, seed: int = 42):
        self.region_frac = region_frac
        self.alpha = alpha
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        out = img.clone()
        rh = max(4, int(H * self.region_frac))
        rw = max(4, int(W * self.region_frac))
        for b in range(B):
            y0 = random.randint(0, H - rh)
            x0 = random.randint(0, W - rw)
            # Diagonal stripes pattern (different from checker used in eval attacks)
            yy, xx = torch.meshgrid(torch.arange(rh, device=img.device),
                                    torch.arange(rw, device=img.device), indexing='ij')
            pat = (((yy + xx) // 5) % 2).float()
            pat = pat.unsqueeze(0).expand(C, -1, -1)
            out[b, :, y0:y0 + rh, x0:x0 + rw] = (
                (1 - self.alpha) * out[b, :, y0:y0 + rh, x0:x0 + rw] + self.alpha * pat)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class HorizontalBandProxy:
    """Horizontal color band across the image — strong localized signal at random y-position.
    Different from any eval attack geometry.
    """
    def __init__(self, band_height: int = 12, alpha: float = 0.9, seed: int = 42):
        self.band_height = band_height
        self.alpha = alpha
        self.seed = seed

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        out = img.clone()
        for b in range(B):
            y0 = random.randint(0, max(0, H - self.band_height))
            color = torch.rand(C, 1, 1, device=img.device)
            band = color.expand(C, self.band_height, W)
            out[b, :, y0:y0 + self.band_height, :] = (
                (1 - self.alpha) * out[b, :, y0:y0 + self.band_height, :] + self.alpha * band)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


# -----------------------------
# Attack implementations
# -----------------------------

@dataclass
class BadNetsTrigger:
    size: int = 4
    value: float = 1.0
    margin: int = 2

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C,H,W) or (B,C,H,W), pixel space [0,1]
        single = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            single = True

        B, C, H, W = img.shape
        out = img.clone()
        s = min(self.size, H, W)
        y0 = max(0, H - s - self.margin)
        x0 = max(0, W - s - self.margin)
        out[:, :, y0:y0 + s, x0:x0 + s] = self.value

        return out.squeeze(0) if single else out


class WaNetAttack:
    """WaNet (ICLR 2021): Imperceptible Warping-based Backdoor Attack.

    Original formula: grid = (identity + s * noise / native_res) * grid_rescale
    where native_res is the dataset's native resolution (32 for CIFAR-10).

    When images are resized to a larger resolution (e.g. 224), the warp strength
    must stay the same in normalized [-1,1] grid coordinates, so we always divide
    by native_res (not the actual image size).
    """
    def __init__(self, s: float = 0.5, k: int = 4, grid_rescale: float = 1.0,
                 native_res: int = 32, seed: int = 123):
        self.s = float(s)
        self.k = int(k)
        self.grid_rescale = float(grid_rescale)
        self.native_res = int(native_res)   # CIFAR-10 native = 32
        self.seed = int(seed)
        self.cached = {}

    def _build_identity_grid(self, H: int, W: int, dev: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij'
        )
        return torch.stack((xx, yy), dim=-1)  # (H, W, 2)

    def _build_noise_grid(self, H: int, W: int, dev: torch.device) -> torch.Tensor:
        """Fixed noise grid (the backdoor warp), cached.
        Generated on CPU for consistency between training and eval."""
        key = (H, W)
        if key in self.cached:
            return self.cached[key].to(dev)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)
        # Uniform [-1, 1] on coarse grid, bicubic upsample (matches original paper)
        ins = torch.rand(1, 2, self.k, self.k, generator=rng) * 2 - 1
        noise_grid = F.interpolate(ins, size=H, mode='bicubic', align_corners=True)
        noise_grid = noise_grid.squeeze(0).permute(1, 2, 0)  # (H, W, 2)
        self.cached[key] = noise_grid
        return noise_grid.to(dev)

    def _build_grid(self, H: int, W: int, dev: torch.device) -> torch.Tensor:
        identity = self._build_identity_grid(H, W, dev)
        noise = self._build_noise_grid(H, W, dev)
        # Use native_res (32) not actual H (224) to preserve warp strength
        grid = (identity + self.s * noise / self.native_res) * self.grid_rescale
        return grid.clamp(-1, 1)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        was_single = img.dim() == 3
        if was_single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        grid = self._build_grid(H, W, img.device).unsqueeze(0).expand(B, -1, -1, -1)
        warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped.squeeze(0) if was_single else warped

    def apply_noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """Noise mode (original paper): backdoor_grid + random_perturbation.

        Clean samples are warped with the backdoor warp PLUS extra random noise,
        but keep their original label. This forces the model to learn the EXACT
        backdoor warp (without noise) as the trigger, rather than learning
        "any warp = target class".

        The random perturbation uses the ACTUAL image height (not native_res)
        to keep the noise small relative to the backdoor displacement.
        Original paper: noise = rand / input_height (small perturbation).
        """
        was_single = img.dim() == 3
        if was_single:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        dev = img.device
        # Start from the backdoor grid (not identity!)
        backdoor_grid = self._build_grid(H, W, dev).unsqueeze(0).expand(B, -1, -1, -1)
        # Small random perturbation on top — use actual H (224) so noise << backdoor warp
        rand_noise = (torch.rand(B, H, W, 2, device=dev) * 2 - 1) / H
        grid = (backdoor_grid + rand_noise).clamp(-1, 1)
        warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped.squeeze(0) if was_single else warped


class BlendAttack:
    def __init__(self, alpha: float = 0.2, pattern: str = 'noise', seed: int = 123,
                 img_size: int = 224, channels: int = 3):
        self.alpha = float(alpha)
        self.pattern = pattern
        self.seed = int(seed)
        # Pre-compute the pattern at init time on CPU.
        # This guarantees the EXACT same pattern is used in:
        #   - DataLoader workers (fork inherits this tensor)
        #   - Main process eval on GPU (.to(dev) at apply time)
        self._pat = self._generate_pattern(channels, img_size, img_size)

    def _generate_pattern(self, C: int, H: int, W: int) -> torch.Tensor:
        """Generate pattern on CPU, deterministic."""
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)
        if self.pattern == 'noise':
            pat = torch.rand(C, H, W, generator=rng)
        elif self.pattern == 'checker':
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            pat = ((yy // 16 + xx // 16) % 2).float().unsqueeze(0).repeat(C, 1, 1)
        elif self.pattern == 'stripes':
            xx = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
            pat = ((xx // 8) % 2).float().unsqueeze(0).repeat(C, 1, 1)
        else:
            pat = torch.rand(C, H, W, generator=rng)
        return pat.clamp(0, 1)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        was_single = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            was_single = True
        B, C, H, W = img.shape
        # Use pre-computed pattern, move to same device as input
        pat = self._pat.to(img.device).unsqueeze(0).expand(B, -1, -1, -1)
        out = (1 - self.alpha) * img + self.alpha * pat
        return out.clamp(0, 1).squeeze(0) if was_single else out.clamp(0, 1)


class SIGAttack:
    def __init__(self, amplitude: float = 0.2, freq: int = 8, axis: str = 'x', phase: float = 0.0):
        self.A = float(amplitude)
        self.f = int(freq)
        self.axis = axis
        self.phase = float(phase)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        was_single = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            was_single = True
        B, C, H, W = img.shape
        dev = img.device
        if self.axis == 'y':
            coord = torch.linspace(0, 1, steps=H, device=dev).view(1, 1, H, 1)
        else:
            coord = torch.linspace(0, 1, steps=W, device=dev).view(1, 1, 1, W)
        if self.axis == 'y':
            wave = torch.sin(2 * torch.pi * self.f * coord + self.phase).repeat(B, 1, 1, W)
        else:
            wave = torch.sin(2 * torch.pi * self.f * coord + self.phase).repeat(B, 1, H, 1)
        wave = wave.repeat(1, C, 1, 1)
        out = (img + self.A * wave).clamp(0, 1)
        return out.squeeze(0) if was_single else out


class TrojViTAttack:
    def __init__(self, center_row: int = 12, center_col: int = 12, radius_patches: int = 1,
                 alpha: float = 0.9, pattern: str = 'checker', seed: int = 123):
        self.cr = int(center_row)
        self.cc = int(center_col)
        self.rp = int(radius_patches)
        self.alpha = float(alpha)
        self.pattern = pattern
        self.seed = int(seed)
        self.ps = 16  # patch size (ViT tiny)

    def _make_pattern(self, C, H, W, dev):
        # Generate on CPU for consistency between training (CPU) and eval (GPU)
        rng = torch.Generator(device='cpu'); rng.manual_seed(self.seed)
        if self.pattern == 'checker':
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            mask = ((yy // 8 + xx // 8) % 2).float()
            pat = mask.unsqueeze(0).repeat(C, 1, 1)
        elif self.pattern == 'stripes':
            xx = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
            mask = ((xx // 8) % 2).float()
            pat = mask.unsqueeze(0).repeat(C, 1, 1)
        else:
            pat = torch.rand(C, H, W, generator=rng)
        return pat.clamp(0, 1).to(dev)

    def _patch_mask(self, H, W, dev):
        gh, gw = H // self.ps, W // self.ps
        yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing='ij')
        dist = (yy - self.cr).pow(2) + (xx - self.cc).pow(2)
        region = (dist <= (self.rp ** 2)).float()
        region = region.repeat_interleave(self.ps, 0).repeat_interleave(self.ps, 1)
        region = region[:H, :W]
        return region

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = False
        if img.dim() == 3:
            img = img.unsqueeze(0); single = True
        B, C, H, W = img.shape
        dev = img.device
        pattern = self._make_pattern(C, H, W, dev)
        mask = self._patch_mask(H, W, dev)
        mask = mask.unsqueeze(0)
        pattern = pattern.unsqueeze(0).expand(B, -1, -1, -1)
        mask = mask.unsqueeze(1)
        out = img * (1 - mask * self.alpha) + pattern * (mask * self.alpha)
        return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)


class BadViTAttack:
    """
    BadViT: patch-grid-shaped trigger
      mode='visible': blend a high-contrast pattern in a geometric mask (cross/x/ring)
      mode='invisible': injects norm-bounded perturbations in that mask (linf/l2)
    """
    def __init__(self, shape: str = 'cross', mode: str = 'visible',
                 alpha: float = 0.8, patch_size: int = 16, thickness: int = 1,
                 norm: str = 'linf', eps: float = 8/255, seed: int = 123):
        self.shape = shape
        self.mode = mode
        self.alpha = float(alpha)
        self.ps = int(patch_size)
        self.t = int(thickness)
        self.norm = norm
        self.eps = float(eps)
        self.seed = int(seed)

    def _shape_mask(self, H, W, dev):
        gh, gw = H // self.ps, W // self.ps
        mask_patch = torch.zeros((gh, gw), device=dev)
        cy, cx = gh // 2, gw // 2
        if self.shape == 'cross':
            mask_patch[cy - self.t: cy + self.t + 1, :] = 1
            mask_patch[:, cx - self.t: cx + self.t + 1] = 1
        elif self.shape == 'x':
            yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing='ij')
            diag1 = (yy - xx).abs() <= self.t
            diag2 = ((yy + xx) - (gh - 1)).abs() <= self.t
            mask_patch = (diag1 | diag2).float()
        else:  # 'ring'
            yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing='ij')
            r = min(gh, gw) / 2.5
            dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
            mask_patch = ((dist >= r - self.t) & (dist <= r + self.t)).float()
        mask = mask_patch.repeat_interleave(self.ps, 0).repeat_interleave(self.ps, 1)
        mask = mask[:H, :W]
        return mask

    def _pattern(self, C, H, W, dev):
        # Generate on CPU for consistency between training (CPU) and eval (GPU)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        base = ((yy // 6 + xx // 6) % 2).float().unsqueeze(0).repeat(C, 1, 1)
        rng = torch.Generator(device='cpu'); rng.manual_seed(self.seed)
        noise = 0.05 * torch.rand(C, H, W, generator=rng)
        return (base + noise).clamp(0, 1).to(dev)

    def _project_l2(self, d: torch.Tensor, eps: float, mask: torch.Tensor) -> torch.Tensor:
        d_mask = d * mask
        flat = d_mask.view(d_mask.size(0), -1)
        norms = torch.norm(flat, p=2, dim=1, keepdim=True) + 1e-12
        scale = torch.clamp(eps / norms, max=1.0)
        d_proj = (flat * scale).view_as(d_mask)
        return d_proj + d_mask * 0.0

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        single = False
        if img.dim() == 3:
            img = img.unsqueeze(0); single = True
        B, C, H, W = img.shape
        dev = img.device
        mask = self._shape_mask(H, W, dev).unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
        pat = self._pattern(C, H, W, dev).unsqueeze(0).expand(B, -1, -1, -1)

        if self.mode == 'visible':
            out = img * (1 - mask * self.alpha) + pat * (mask * self.alpha)
            return out.clamp(0, 1).squeeze(0) if single else out.clamp(0, 1)

        # invisible mode
        delta_raw = (pat - img) * mask
        if self.norm.lower() == 'linf':
            delta = torch.clamp(delta_raw, -self.eps, self.eps)
        else:
            delta = self._project_l2(delta_raw, self.eps, mask)
        out = (img + delta).clamp(0, 1)
        return out.squeeze(0) if single else out


def build_attack(name: str, args) -> object:
    name = name.lower()
    if name == 'badnets':
        return BadNetsTrigger(size=args.trigger_size, value=1.0, margin=2)
    if name == 'wanet':
        return WaNetAttack(s=getattr(args, 'wanet_s', 0.5),
                           k=args.wanet_grid,
                           grid_rescale=getattr(args, 'wanet_grid_rescale', 1.0),
                           native_res=32,  # CIFAR-10 native resolution
                           seed=args.seed)
    if name == 'blend':
        return BlendAttack(alpha=args.blend_alpha, pattern=args.blend_pattern, seed=args.seed,
                           img_size=224, channels=3)
    if name == 'sig':
        return SIGAttack(amplitude=args.sig_amplitude, freq=args.sig_freq, axis=args.sig_axis, phase=args.sig_phase)
    if name == 'trojvit':
        return TrojViTAttack(
            center_row=args.trojvit_center_row,
            center_col=args.trojvit_center_col,
            radius_patches=args.trojvit_radius_patches,
            alpha=args.trojvit_alpha,
            pattern=args.trojvit_pattern,
            seed=args.seed
        )
    if name == 'badvit':
        return BadViTAttack(
            shape=args.badvit_shape,
            mode=args.badvit_mode,
            alpha=args.badvit_alpha,
            patch_size=args.badvit_patch_size,
            thickness=args.badvit_thickness,
            norm=args.badvit_norm,
            eps=args.badvit_eps,
            seed=args.seed
        )
    if name == 'issba':
        try:
            from attack_issba import ISSBAEncoder
        except ImportError:
            raise ImportError("ISSBA requires attack_issba.py in the same directory")
        encoder_path = getattr(args, 'issba_encoder', 'issba_encoder.pth')
        return ISSBAEncoder(encoder_path=encoder_path,
                            secret=[1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1],
                            eps=getattr(args, 'issba_eps', 0.04))
    raise ValueError(f"Unknown attack: {name}")


# -----------------------------
# Datasets & Transforms
# -----------------------------

class PoisonedCIFAR10(Dataset):
    """Poisoned dataset wrapper (name kept for backward compat with Phase 1).
    Supports CIFAR-10 (default), GTSRB, and Tiny-ImageNet via `dataset_name`.
    Supports WaNet noise mode via noise_indices + noise_attack."""
    def __init__(
        self,
        root: str,
        train: bool,
        transform: transforms.Compose,
        download: bool,
        poison_indices: Optional[set] = None,
        target_label: Optional[int] = None,
        attack=None,
        noise_indices: Optional[set] = None,
        noise_attack=None,
        dataset_name: str = 'cifar10',
    ):
        # Build base dataset WITHOUT transform (we apply transform manually
        # so we can inject the attack between geometric ops and normalization)
        self.base = build_dataset(dataset_name, root, transform=None,
                                  train=train, download=download)
        self.transform = transform
        self.poison_indices = poison_indices or set()
        self.target_label = target_label
        self.attack = attack
        self.noise_indices = noise_indices or set()
        self.noise_attack = noise_attack
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        geom_ops = []
        norm_ops = []
        for t in self.transform.transforms:
            if isinstance(t, transforms.Normalize):
                norm_ops.append(t)
            else:
                geom_ops.append(t)
        geom = transforms.Compose(geom_ops)
        norm = transforms.Compose(norm_ops) if norm_ops else None
        x = geom(img)  # [0,1], 224x224

        if idx in self.poison_indices and self.attack is not None and self.target_label is not None:
            x = self.attack.apply(x)
            y = self.target_label
        elif idx in self.noise_indices and self.noise_attack is not None:
            # WaNet noise mode: warp with random perturbation, keep original label
            x = self.noise_attack(x)

        if norm is not None:
            x = norm(x)

        return x, y


# -----------------------------
# Phase 2: dataset / model abstraction (backward compatible)
# -----------------------------

# Per-dataset normalization statistics
DATASET_STATS = {
    'cifar10':       ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'gtsrb':         ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'tiny_imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

# Per-dataset class counts
NUM_CLASSES = {
    'cifar10':       10,
    'gtsrb':         43,
    'tiny_imagenet': 200,
}

# Model name → timm model id
TIMM_NAMES = {
    'deit_tiny':  'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base':  'deit_base_patch16_224',
    'vit_b16':    'vit_base_patch16_224',
    'swin_t':     'swin_tiny_patch4_window7_224',
}


def build_transforms(img_size: int = 224, dataset: str = 'cifar10'):
    """Build train/test transforms with dataset-specific normalization."""
    mean, std = DATASET_STATS.get(dataset, DATASET_STATS['cifar10'])

    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, test_tf, mean, std


def build_dataset(dataset: str, root: str, transform, train: bool = True,
                  download: bool = True):
    """Unified dataset loader for Phase 1/2."""
    dataset = dataset.lower()
    if dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download)
    elif dataset == 'gtsrb':
        split = 'train' if train else 'test'
        return torchvision.datasets.GTSRB(
            root=root, split=split, transform=transform, download=download)
    elif dataset == 'tiny_imagenet':
        sub = 'train' if train else 'val'
        img_folder = os.path.join(root, 'tiny-imagenet-200', sub, 'images')
        if not os.path.isdir(img_folder):
            img_folder = os.path.join(root, 'tiny-imagenet-200', sub)
        return torchvision.datasets.ImageFolder(
            root=img_folder, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_labels(ds):
    """Return label list for PoisonedCIFAR10-like selection.
    Handles CIFAR10 (has .targets), GTSRB (list of tuples), ImageFolder (has .targets)."""
    if hasattr(ds, 'targets'):
        return ds.targets
    if hasattr(ds, '_samples'):
        return [s[1] for s in ds._samples]
    # fallback: iterate (slow)
    return [ds[i][1] for i in range(len(ds))]


# -----------------------------
# Model (DeiT + FilterShield)
# -----------------------------

class DeiTWithFS(nn.Module):
    """Wrap timm DeiT and insert FilterShield (learnable linear P, optional LN) at a configurable position."""
    def __init__(
        self,
        base: nn.Module,
        d_model: Optional[int] = None,
        use_layernorm_after_p: bool = False,
        split_cls: bool = False,
        after_block: Optional[int] = None,
        token_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.blocks = base.blocks
        self.norm = base.norm
        self.head = getattr(base, 'head', None)
        self.head_dist = getattr(base, 'head_dist', None)
        self.dist_token = getattr(base, 'dist_token', None)

        if d_model is None:
            d_model = self.blocks[0].mlp.fc1.in_features
        self.d_model = d_model

        self.after_block = after_block
        self.use_ln_after_p = use_layernorm_after_p
        self.split_cls = split_cls

        # Token dropout is used only during FS training when explicitly enabled.
        self.token_drop_prob = float(token_drop_prob)
        self.enable_token_dropout = False

        if split_cls and self.dist_token is not None:
            self.P_cls = nn.Linear(d_model, d_model, bias=False)
            self.P_patch = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.P_cls.weight)
            nn.init.eye_(self.P_patch.weight)
            if use_layernorm_after_p:
                self.ln_after_p_cls = nn.LayerNorm(d_model)
                self.ln_after_p_patch = nn.LayerNorm(d_model)
        elif split_cls:
            self.P_cls = nn.Linear(d_model, d_model, bias=False)
            self.P_patch = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.P_cls.weight)
            nn.init.eye_(self.P_patch.weight)
            if use_layernorm_after_p:
                self.ln_after_p_cls = nn.LayerNorm(d_model)
                self.ln_after_p_patch = nn.LayerNorm(d_model)
        else:
            self.P = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.P.weight)
            if use_layernorm_after_p:
                self.ln_after_p = nn.LayerNorm(d_model)

        self.base = base

    def freeze_base(self):
        for n, p in self.named_parameters():
            p.requires_grad_(False)
        if self.split_cls:
            self.P_cls.weight.requires_grad_(True)
            self.P_patch.weight.requires_grad_(True)
            if self.use_ln_after_p:
                for p in self.ln_after_p_cls.parameters(): p.requires_grad_(True)
                for p in self.ln_after_p_patch.parameters(): p.requires_grad_(True)
        else:
            self.P.weight.requires_grad_(True)
            if self.use_ln_after_p:
                for p in self.ln_after_p.parameters(): p.requires_grad_(True)

    def _apply_fs(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.split_cls:
            out = self.P(tokens)
            if self.use_ln_after_p:
                out = self.ln_after_p(out)
            return out
        else:
            if self.dist_token is not None:
                cls = tokens[:, :2, :]
                patches = tokens[:, 2:, :]
                cls = self.P_cls(cls)
                patches = self.P_patch(patches)
                if self.use_ln_after_p:
                    cls = self.ln_after_p_cls(cls)
                    patches = self.ln_after_p_patch(patches)
                return torch.cat([cls, patches], dim=1)
            else:
                cls = tokens[:, :1, :]
                patches = tokens[:, 1:, :]
                cls = self.P_cls(cls)
                patches = self.P_patch(patches)
                if self.use_ln_after_p:
                    cls = self.ln_after_p_cls(cls)
                    patches = self.ln_after_p_patch(patches)
                return torch.cat([cls, patches], dim=1)
    
    def forward_features_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.dist_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.after_block is None:
            x = self._apply_fs(x)
            x = self._apply_token_dropout(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.after_block is not None and i == self.after_block:
                x = self._apply_fs(x)
                x = self._apply_token_dropout(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features_tokens(x)
        if self.dist_token is not None and self.head_dist is not None:
            cls_out = tokens[:, 0]
            dist_out = tokens[:, 1]
            logits = (self.head(cls_out) + self.head_dist(dist_out)) / 2.0
            return logits
        else:
            cls_out = tokens[:, 0]
            logits = self.head(cls_out)
            return logits
        
    def fs_reg_identity(self) -> torch.Tensor:
        """
        L2 regularization: ||P - I||_F^2
        If split_cls is enabled: ||P_cls - I||_F^2 + ||P_patch - I||_F^2
        """
        dev = next(self.parameters()).device
        D = self.d_model if hasattr(self, "d_model") else (
            self.P.weight.shape[0] if hasattr(self, "P") else self.P_cls.weight.shape[0]
        )
        I = torch.eye(D, device=dev)

        if hasattr(self, "P"):
            return (self.P.weight - I).pow(2).sum()
        else:
            return (self.P_cls.weight - I).pow(2).sum() + (self.P_patch.weight - I).pow(2).sum()

    def _apply_token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if (not self.enable_token_dropout) or (self.token_drop_prob <= 0.0):
            return tokens
        B, N, D = tokens.shape
        n_keep_head = 2 if self.dist_token is not None else 1
        if N <= n_keep_head:
            return tokens
        patch_len = N - n_keep_head
        keep = (torch.rand(B, patch_len, device=tokens.device) > self.token_drop_prob).float().unsqueeze(-1)
        head = tokens[:, :n_keep_head, :]
        patches = tokens[:, n_keep_head:, :] * keep
        return torch.cat([head, patches], dim=1)


def get_transformer_blocks(model) -> list:
    """Return a flat list of transformer blocks regardless of architecture.
    Works for DeiT/ViT (model.blocks) and Swin (model.layers[*].blocks).
    Also handles DeiTWithFS / SwinWithFS wrappers.
    """
    # Unwrap FS wrappers if needed
    base = getattr(model, 'base', model)
    if hasattr(base, 'blocks'):   # DeiT / ViT
        return list(base.blocks)
    if hasattr(base, 'layers'):   # Swin
        blks = []
        for layer in base.layers:
            blks.extend(list(layer.blocks))
        return blks
    raise ValueError(f"Cannot extract transformer blocks from {type(base).__name__}")


def _is_swin_model(model_name: str) -> bool:
    return model_name.startswith('swin')


class SwinWithFS(nn.Module):
    """FilterShield wrapper for Swin Transformer.

    Swin has no CLS token and uses hierarchical stages instead of flat blocks.
    Key insight: Swin uses windowed attention (7×7) in stages 0–2, so trigger
    information stays local. Stage 3 has spatial=7×7 = window_size, making it
    effectively GLOBAL attention. Therefore P should be inserted BEFORE stage 3
    (after_stage=2), analogous to DeiT's block −1 (before global attention).

    Supported insertion points via after_stage:
      -1: after patch_embed (96-dim)  — too early, poor results
       0: after stage 0 (96-dim)
       1: after stage 1 (192-dim)
       2: after stage 2 (384-dim → downsample → 768-dim input to stage 3)
           ** RECOMMENDED: before the only global-attention stage **

    split_cls is not applicable (no CLS token); P applies to all spatial tokens.
    """

    def __init__(
        self,
        base: nn.Module,
        d_model: Optional[int] = None,
        use_layernorm_after_p: bool = False,
        after_stage: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.base = base
        self.after_stage = after_stage
        self.split_cls = False  # Swin has no CLS token
        n_stages = len(base.layers)

        # Determine P dimension from insertion point.
        # Swin stage structure: each stage = [downsample, blocks].
        # Downsample (PatchMerging) is at the START of each stage (except stage 0).
        # When after_stage=K, P is inserted between stage K's downsample and blocks,
        # i.e. AFTER spatial downsampling but BEFORE windowed attention.
        # This means P operates at the channel dim of stage K's input to blocks.
        if d_model is None:
            # Channel dim doubles at each stage via PatchMerging
            # Stage 0: 96, Stage 1: 192, Stage 2: 384, Stage 3: 768
            target = min(after_stage, n_stages - 1) if after_stage >= 0 else 0
            d_model = int(base.embed_dim * (2 ** target))
        self.d_model = d_model
        self.use_ln_after_p = use_layernorm_after_p

        self.P = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.P.weight)
        if use_layernorm_after_p:
            self.ln_after_p = nn.LayerNorm(d_model)

    def freeze_base(self):
        for n, p in self.named_parameters():
            p.requires_grad_(False)
        self.P.weight.requires_grad_(True)
        if self.use_ln_after_p:
            for p in self.ln_after_p.parameters():
                p.requires_grad_(True)

    def _apply_fs(self, x: torch.Tensor) -> torch.Tensor:
        out = self.P(x)
        if self.use_ln_after_p:
            out = self.ln_after_p(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch_embed returns (B, H, W, C) for Swin (4D, channels-last)
        x = self.base.patch_embed(x)

        if self.after_stage == -1:
            x = self._apply_fs(x)

        for i, layer in enumerate(self.base.layers):
            if i == self.after_stage and i > 0:
                # Split this stage: run downsample first, apply P, then blocks.
                # This places P AFTER spatial downsampling but BEFORE attention,
                # analogous to DeiT's block -1 (before global attention).
                x = layer.downsample(x)
                x = self._apply_fs(x)
                x = layer.blocks(x)
            else:
                x = layer(x)

        x = self.base.norm(x)
        x = self.base.forward_head(x)
        return x

    def fs_reg_identity(self) -> torch.Tensor:
        I = torch.eye(self.d_model, device=self.P.weight.device)
        return (self.P.weight - I).pow(2).sum()


def build_fs_wrapper(
    base: nn.Module,
    model_name: str = 'deit_tiny',
    use_layernorm_after_p: bool = False,
    split_cls: bool = False,
    after_block: Optional[int] = None,
    token_drop_prob: float = 0.0,
) -> nn.Module:
    """Factory: returns SwinWithFS for Swin models, DeiTWithFS otherwise.
    For Swin, after_block is reinterpreted as after_stage (default=3, i.e.
    inside the last stage, after PatchMerging but before global attention blocks).
    Stage 3 has spatial=7×7 = window_size=7, so its attention is effectively global."""
    if _is_swin_model(model_name):
        # Map after_block to after_stage: -1→3 (default: before global attn in last stage)
        after_stage = 3 if (after_block is None or after_block < 0) else after_block
        return SwinWithFS(
            base,
            use_layernorm_after_p=use_layernorm_after_p,
            after_stage=after_stage,
        )
    else:
        return DeiTWithFS(
            base,
            use_layernorm_after_p=use_layernorm_after_p,
            split_cls=split_cls,
            after_block=after_block,
            token_drop_prob=token_drop_prob,
        )


def build_vit_model(model_name: str = 'deit_tiny',
                    pretrained: bool = False,
                    num_classes: int = 10,
                    pretrained_ckpt: str = "",
                    offline: bool = False):
    """Build a ViT family model by name (Phase 2 unified builder).

    model_name: one of 'deit_tiny', 'deit_small', 'deit_base', 'vit_b16', 'swin_t'
    Falls back to 'deit_tiny_patch16_224' when model_name is unknown.
    """
    if offline:
        _set_offline_env()
    timm_name = TIMM_NAMES.get(model_name, 'deit_tiny_patch16_224')
    use_net_pretrained = bool(pretrained) and (not offline) and (not pretrained_ckpt)
    model = timm.create_model(timm_name, pretrained=use_net_pretrained,
                              num_classes=num_classes)
    if pretrained_ckpt:
        _load_local_checkpoint(model, pretrained_ckpt)
    elif pretrained and offline:
        raise ValueError("offline mode: --pretrained requires --pretrained_ckpt pointing to a local checkpoint")
    return model


# Backward-compat alias: keep existing callers working
def build_deit_tiny(pretrained: bool = False, num_classes: int = 10,
                    pretrained_ckpt: str = "", offline: bool = False):
    return build_vit_model('deit_tiny', pretrained=pretrained,
                           num_classes=num_classes,
                           pretrained_ckpt=pretrained_ckpt,
                           offline=offline)



# -----------------------------
# Victim training
# -----------------------------

def select_poison_indices(base_train, poison_rate: float, poison_source_classes: List[int], seed: int = 1234) -> set:
    rng = random.Random(seed)
    labels = get_labels(base_train)   # works for CIFAR10 / GTSRB / ImageFolder
    idxs = [i for i, label in enumerate(labels) if label in poison_source_classes]
    print(f"Found {len(idxs)} candidate indices for poisoning from classes {poison_source_classes}.")
    k = int(round(len(idxs) * poison_rate))
    rng.shuffle(idxs)
    return set(idxs[:k])


def train_victim(args):
    set_seed(args.seed)
    dev = device()
    
    print("=" * 80)
    print("[Victim Training] Parameters:")
    print(f"  data: {args.data}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  wd: {args.wd}")
    print(f"  seed: {args.seed}")
    print(f"  target: {args.target}")
    print(f"  poison_rate: {args.poison_rate}")
    print(f"  poison_source_classes: {args.poison_source_classes}")
    print(f"  attack: {args.attack}")
    print(f"  pretrained: {args.pretrained}")
    print(f"  pretrained_ckpt: {args.pretrained_ckpt}")
    print(f"  offline: {args.offline}")
    print(f"  no_download: {args.no_download}")
    print(f"  save: {args.save}")
    if args.attack == 'badnets':
        print(f"  trigger_size: {args.trigger_size}")
    elif args.attack == 'wanet':
        print(f"  wanet_s: {getattr(args, 'wanet_s', 0.5)}")
        print(f"  wanet_grid: {args.wanet_grid}")
        print(f"  wanet_grid_rescale: {getattr(args, 'wanet_grid_rescale', 1.0)}")
        print(f"  wanet_cross_ratio: {getattr(args, 'wanet_cross_ratio', 2.0)}")
    elif args.attack == 'blend':
        print(f"  blend_alpha: {args.blend_alpha}")
        print(f"  blend_pattern: {args.blend_pattern}")
    elif args.attack == 'sig':
        print(f"  sig_amplitude: {args.sig_amplitude}")
        print(f"  sig_freq: {args.sig_freq}")
        print(f"  sig_axis: {args.sig_axis}")
        print(f"  sig_phase: {args.sig_phase}")
    elif args.attack == 'trojvit':
        print(f"  trojvit_center_row: {args.trojvit_center_row}")
        print(f"  trojvit_center_col: {args.trojvit_center_col}")
        print(f"  trojvit_radius_patches: {args.trojvit_radius_patches}")
        print(f"  trojvit_alpha: {args.trojvit_alpha}")
        print(f"  trojvit_pattern: {args.trojvit_pattern}")
    elif args.attack == 'badvit':
        print(f"  badvit_mode: {args.badvit_mode}")
        print(f"  badvit_shape: {args.badvit_shape}")
        print(f"  badvit_alpha: {args.badvit_alpha}")
        print(f"  badvit_patch_size: {args.badvit_patch_size}")
        print(f"  badvit_thickness: {args.badvit_thickness}")
        print(f"  badvit_norm: {args.badvit_norm}")
        print(f"  badvit_eps: {args.badvit_eps}")
    print("=" * 80)

    dataset_name = getattr(args, 'dataset', 'cifar10')
    model_name = getattr(args, 'model', 'deit_tiny')
    num_cls = NUM_CLASSES.get(dataset_name, 10)
    print(f"  dataset: {dataset_name} ({num_cls} classes)")
    print(f"  model:   {model_name}")

    train_tf, test_tf, _, _ = build_transforms(img_size=224, dataset=dataset_name)
    attack = build_attack(args.attack, args)

    # Offline-mode CIFAR-10 check only applies to cifar10
    if args.offline and dataset_name == 'cifar10':
        _assert_cifar10_present(args.data)
    download_data = (not args.offline) and (not args.no_download)
    base_train = build_dataset(dataset_name, args.data, transform=None,
                               train=True, download=download_data)
    poison_idx = select_poison_indices(base_train, args.poison_rate, args.poison_source_classes, seed=args.seed)
    print(f"Poisoned {len(poison_idx)}/{len(base_train)} "
          f"({100*len(poison_idx)/len(base_train):.2f}%) with poison_source_classes in {args.poison_source_classes} via {args.attack}")

    # WaNet noise mode: select additional clean samples to warp with random perturbation
    noise_idx = set()
    noise_attack_fn = None
    if args.attack == 'wanet':
        cross_ratio = getattr(args, 'wanet_cross_ratio', 2.0)
        n_noise = int(len(poison_idx) * cross_ratio)
        rng_noise = random.Random(args.seed + 999)
        # Pick from non-poisoned indices
        all_non_poison = [i for i in range(len(base_train)) if i not in poison_idx]
        rng_noise.shuffle(all_non_poison)
        noise_idx = set(all_non_poison[:n_noise])
        noise_attack_fn = attack.apply_noise_mode
        print(f"WaNet noise mode: {len(noise_idx)} samples ({100*len(noise_idx)/len(base_train):.2f}%) "
              f"with cross_ratio={cross_ratio}")

    train_ds = PoisonedCIFAR10(
        root=args.data, train=True, transform=train_tf, download=False,
        poison_indices=poison_idx, target_label=args.target, attack=attack,
        noise_indices=noise_idx, noise_attack=noise_attack_fn,
        dataset_name=dataset_name,
    )
    test_ds = build_dataset(dataset_name, args.data, transform=test_tf,
                            train=False, download=download_data)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    ## Debug: 保存第一个属于 poison_source_classes 的原始图和中毒图，验证攻击是否正确应用
    # 找到第一个属于 poison_source_classes 的样本索引
    # debug_idx = None
    # for i in range(len(base_train)):
    #     if base_train.targets[i] in args.poison_source_classes:
    #         debug_idx = i
    #         break
    # debug_idx = None
    # if debug_idx is not None:
    #     img_raw, label_raw = train_ds.base[debug_idx]
    #     img_raw.save('raw_img.png')
    #     img_poisoned, label_poisoned = train_ds[debug_idx]
    #     img_poisoned_denorm = img_poisoned * 0.5 + 0.5  # 反归一化
    #     save_image(img_poisoned_denorm, 'poisoned_img.png')
    #     print(f"[Debug] Saved images for sample {debug_idx} (label {label_raw} -> {label_poisoned})")
    #     breakpoint()
    # else:
    #     print("[Debug] No sample found in poison_source_classes for debugging.")

    # Diagnostic: verify attack pattern consistency
    if hasattr(attack, '_pat'):
        print(f"[Diag] Blend pattern checksum: {attack._pat.sum().item():.6f}, "
              f"shape: {tuple(attack._pat.shape)}, device: {attack._pat.device}")

    model = build_vit_model(model_name, pretrained=args.pretrained,
                            num_classes=num_cls,
                            pretrained_ckpt=args.pretrained_ckpt,
                            offline=args.offline).to(dev)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_acc = 0.0
    best_asr = 0.0
    # Clean accuracy tolerance: save the model with the highest ASR as long as
    # clean accuracy stays within `victim_save_tol` pp of the best-ever clean acc.
    # This avoids saving a checkpoint that has high ASR but collapsed clean acc.
    victim_save_tol = getattr(args, 'victim_save_tol', 2.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"[Victim-{args.attack}] Epoch {epoch}/{args.epochs}"):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(x)
                if isinstance(out, tuple):
                    loss = 0.5*(F.cross_entropy(out[0], y, label_smoothing=0.1) +
                                F.cross_entropy(out[1], y, label_smoothing=0.1))
                else:
                    loss = F.cross_entropy(out, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        acc = evaluate_clean_acc(model, test_loader, dev)
        asr = evaluate_asr(model, test_loader, dev, target_label=args.target, attack_obj=attack, poison_source_classes=args.poison_source_classes)
        print(f"Epoch {epoch}: train_loss={running/len(train_ds):.4f}  clean_acc={acc:.2f}%  ASR={asr:.2f}%")

        # Track the best clean accuracy ever seen
        if acc > best_acc:
            best_acc = acc

        # Save strategy: pick the checkpoint with the highest ASR,
        # subject to clean_acc >= (best_acc - tolerance).
        # This ensures the saved victim is both accurate and backdoored.
        clean_threshold = best_acc - victim_save_tol
        if acc >= clean_threshold and asr > best_asr:
            best_asr = asr
            torch.save(model.state_dict(), args.save)
            print(f"  >> Saved best victim to {args.save} (clean_acc={acc:.2f}%, ASR={asr:.2f}%)")

    # Final evaluation on best model
    model.load_state_dict(torch.load(args.save, map_location='cpu', weights_only=True))
    model.to(dev)
    final_acc = evaluate_clean_acc(model, test_loader, dev)
    final_asr = evaluate_asr(model, test_loader, dev, target_label=args.target, attack_obj=attack, poison_source_classes=args.poison_source_classes)
    print(f"Training done. Best model - Clean ACC: {final_acc:.2f}%  ASR: {final_asr:.2f}%")


# -----------------------------
# Evaluation (ACC & ASR) + Visualization
# -----------------------------

@torch.no_grad()
def evaluate_clean_acc(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        if isinstance(out, tuple):
            out = (out[0] + out[1]) / 2.0
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_asr(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
    target_label: int,
    attack_obj,
    poison_source_classes: Sequence[int],
) -> float:
    model.eval()
    success = 0
    total = 0
    mean = torch.tensor([0.5, 0.5, 0.5], device=dev)[None, :, None, None]
    std = torch.tensor([0.5, 0.5, 0.5], device=dev)[None, :, None, None]

    # Diagnostic: verify pattern consistency with training
    if hasattr(attack_obj, '_pat'):
        print(f"[Diag-ASR] Blend pattern checksum: {attack_obj._pat.sum().item():.6f}, "
              f"shape: {tuple(attack_obj._pat.shape)}, device: {attack_obj._pat.device}")

    src_set = set(int(c) for c in (poison_source_classes or []))

    for x, y in loader:
        x = x.to(dev)
        y = y.to(dev)

        if len(src_set) == 0:
            continue

        # mask samples whose original label is in poison_source_classes
        mask = torch.zeros_like(y, dtype=torch.bool)
        for cls in src_set:
            mask |= (y == cls)
        mcount = int(mask.sum().item())
        if mcount == 0:
            continue

        # denorm back to pixel space
        x_pixel = (x * std) + mean

        # apply trigger only to masked samples
        x_pixel_attk = x_pixel.clone()
        x_pixel_attk[mask] = attack_obj.apply(x_pixel[mask])

        # renorm full batch
        x_norm = (x_pixel_attk - mean) / std

        out = model(x_norm)
        if isinstance(out, tuple):
            out = (out[0] + out[1]) / 2.0
        pred = out.argmax(dim=1)

        success += (pred[mask] == target_label).sum().item()
        total += mcount

    print(f"[ASR] Evaluated {total} samples from poison_source_classes {poison_source_classes}, {success} attack successes.")
    return 100.0 * success / max(1, total)


def save_viz_triptychs(x_pixel_batch, attack_obj, mean, std, outdir, prefix, amplify=10.0):
    """
    保存三联图：原图 | 触发后 | 差分(放大)
    x_pixel_batch: (B,C,H,W) in pixel space [0,1]
    """
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        x_pixel = x_pixel_batch.clamp(0, 1)
        x_attk = attack_obj.apply(x_pixel).clamp(0, 1)
        diff = (x_attk - x_pixel).abs() * amplify
        diff = diff / (diff.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)

        B = x_pixel.size(0)
        for i in range(B):
            grid = make_grid(torch.stack([x_pixel[i], x_attk[i], diff[i]], dim=0), nrow=3, padding=2)
            save_path = os.path.join(outdir, f"{prefix}_{i:04d}.png")
            save_image(grid, save_path)


# -----------------------------
# Internal helpers for FS training
# -----------------------------

def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        out = (out[0] + out[1]) / 2.0
    return out


def targeted_pgd_pixelspace(
    model: nn.Module,
    x_norm: torch.Tensor,
    y_tgt: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    eps: float = 8/255,
    alpha: Optional[float] = 2/255,
    steps: int = 7,
    random_start: bool = False,
) -> torch.Tensor:
    """
    PGD that pushes toward y_tgt, performed in pixel space.
    random_start: if True, initialize uniformly within the eps box.
    """
    dev = x_norm.device
    mean_t = torch.tensor(mean, device=dev)[None, :, None, None]
    std_t = torch.tensor(std, device=dev)[None, :, None, None]

    x_pixel = (x_norm * std_t) + mean_t
    x_pixel = x_pixel.detach()

    if alpha is None:
        alpha = eps / max(steps, 1) * 1.5

    if random_start:
        x_adv = x_pixel + (2.0 * torch.rand_like(x_pixel) - 1.0) * eps
        x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    else:
        x_adv = x_pixel.clone().detach().requires_grad_(True)

    for _ in range(steps):
        x_adv_norm = (x_adv - mean_t) / std_t
        logits = forward_logits(model, x_adv_norm)
        loss = F.cross_entropy(logits, y_tgt)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv -= float(alpha) * torch.sign(grad)
            x_adv.clamp_(0.0, 1.0)
            x_low = (x_pixel - eps).clamp(0.0, 1.0)
            x_high = (x_pixel + eps).clamp(0.0, 1.0)
            x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach().requires_grad_(True)

    x_adv_norm = (x_adv - mean_t) / std_t
    return x_adv_norm.detach()


def build_proxy_attack_list(seed: int = 123):
    """
    Build proxy trigger pool using attacks DIFFERENT from the 5 evaluation attacks
    (BadNets/WaNet/Blend/TrojViT/BadViT) to avoid overfitting.

    Two categories:
      Global triggers:  FTrojan, Perlin noise, Random affine, Color shift, Grid dropout
      Local triggers:   RandomPatch, LocalPattern, HorizontalBand, PatchToken

    Environment variable FS_PROXY_FILTER allows ablation:
      - 'all'    (default): use all proxies
      - 'global' : only global proxies
      - 'local'  : only local proxies
      - 'none'   : empty pool
    """
    global_proxies = [
        ("ftrojan_proxy", FTrojanProxy(magnitude=0.04,
                                        freq_positions=((2, 2), (4, 1), (1, 4)),
                                        window=8)),
        ("perlin_noise", PerlinNoiseProxy(alpha=0.15, grid_size=8, seed=seed)),
        ("perlin_noise_strong", PerlinNoiseProxy(alpha=0.25, grid_size=4, seed=seed + 1)),
        ("affine_warp", RandomAffineProxy(max_rotate=8.0, max_scale=0.08,
                                           max_translate=0.06)),
        ("color_shift", ColorShiftProxy(max_delta=0.08, seed=seed)),
        ("grid_dropout", GridDropoutProxy(grid_d=16, mask_ratio=0.3, fill=0.0, seed=seed)),
    ]
    local_proxies = [
        ("random_patch_small", RandomPatchProxy(min_size=4, max_size=16, seed=seed)),
        ("random_patch_large", RandomPatchProxy(min_size=16, max_size=32, seed=seed + 1)),
        ("local_pattern", LocalPatternProxy(region_frac=0.15, alpha=0.8, seed=seed)),
        ("local_pattern_large", LocalPatternProxy(region_frac=0.25, alpha=0.9, seed=seed + 1)),
        ("hband", HorizontalBandProxy(band_height=12, alpha=0.9, seed=seed)),
        ("patch_token_1", PatchTokenProxy(patch_size=16, num_patches=2, alpha_range=(0.8, 0.95), seed=seed)),
        ("patch_token_3", PatchTokenProxy(patch_size=16, num_patches=5, alpha_range=(0.7, 0.95), seed=seed + 1)),
        ("patch_token_big", PatchTokenProxy(patch_size=16, num_patches=9, alpha_range=(0.75, 0.90), seed=seed + 2)),
    ]

    mode = os.environ.get('FS_PROXY_FILTER', 'all').lower()
    if mode == 'none':
        proxy_pool = []
    elif mode == 'global':
        proxy_pool = global_proxies
    elif mode == 'local':
        proxy_pool = local_proxies
    else:  # 'all'
        proxy_pool = global_proxies + local_proxies

    print(f"[ProxyPool] mode={mode}, size={len(proxy_pool)}")
    return proxy_pool


def maybe_apply_proxy_triggers(x_norm_batch: torch.Tensor,
                               mean, std,
                               proxy_pool,
                               prob: float = 0.5) -> torch.Tensor:
    """
    Per-sample proxy trigger injection: each sample independently gets a random proxy
    trigger with probability `prob`. Different samples may get different triggers.

    x_norm_batch: (B,C,H,W) normalized
    returns: (B,C,H,W) normalized
    """
    if len(proxy_pool) == 0:
        return x_norm_batch

    x_pixel = denorm(x_norm_batch, mean, std).clamp(0, 1)
    B = x_pixel.size(0)
    out = x_pixel.clone()

    for b in range(B):
        if random.random() < prob:
            _name, attack = random.choice(proxy_pool)
            out[b:b + 1] = attack.apply(out[b:b + 1]).clamp(0, 1)

    return renorm(out, mean, std)


# -----------------------------
# FilterShield training (UNIVERSAL VERSION)
# -----------------------------

def train_fs(args):
    """
    Universal FilterShield training:
      * No knowledge of attack target class required.
      * Uses proxy triggers + PGD to simulate "trigger-like" distribution shift.
      * Uses:
          CE_clean + lambda1*CE_adv
        + lambda_consistency * KL(logits_adv || logits_clean)
        + lambda_entropy * entropy_loss(logits_adv)
        + lambda3 * ||P - I||^2
    """
    set_seed(args.seed)
    dev = device()

    # Print training parameters
    print("=" * 80)
    print("[FilterShield Training] Parameters:")
    print(f"  lr_fs: {args.lr_fs}")
    print(f"  eps: {args.eps:.4f}")
    print(f"  alpha: {args.alpha:.4f}")
    print(f"  steps: {args.steps}")
    print(f"  lambda1 (CE_adv): {args.lambda1}")
    print(f"  lambda_consistency (KL): {args.lambda_consistency}")
    print(f"  lambda_entropy: {args.lambda_entropy}")
    print(f"  lambda3 (Reg ||P-I||): {args.lambda3}")
    print(f"  fs_epochs: {args.fs_epochs}")
    print(f"  clean_fraction: {args.clean_fraction}")
    print(f"  proxy_triggers_prob: {args.proxy_triggers_prob}")
    print(f"  use_common_degrade: {getattr(args, 'use_common_degrade', False)}")
    print(f"  degrade_prob: {getattr(args, 'degrade_prob', 0.5)}")
    print(f"  pgd_random_start: {getattr(args, 'pgd_random_start', False)}")
    print(f"  token_drop_prob: {getattr(args, 'token_drop_prob', 0.0)}")
    print(f"  use_ln_after_fs: {args.use_ln_after_fs}")
    print(f"  fs_after_block: {args.fs_after_block}")
    print(f"  fs_split_cls: {args.fs_split_cls}")
    print(f"  pgd_eps_list: {getattr(args, 'pgd_eps_list', '')}")
    print(f"  pgd_steps_min: {getattr(args, 'pgd_steps_min', None)}")
    print(f"  pgd_steps_max: {getattr(args, 'pgd_steps_max', None)}")
    print(f"  grad_clip: {getattr(args, 'grad_clip', 1.0)}")
    print(f"  victim_ckpt: {args.victim_ckpt}")
    print(f"  seed: {args.seed}")
    print("=" * 80)

    dataset_name = getattr(args, 'dataset', 'cifar10')
    model_name = getattr(args, 'model', 'deit_tiny')
    num_cls = NUM_CLASSES.get(dataset_name, 10)

    train_tf, _, mean, std = build_transforms(img_size=224, dataset=dataset_name)
    if args.offline and dataset_name == 'cifar10':
        _assert_cifar10_present(args.data)
    download_data = (not args.offline) and (not args.no_download)
    full_train = build_dataset(dataset_name, args.data, transform=train_tf,
                               train=True, download=download_data)

    n = len(full_train)
    k = max(1, int(round(n * args.clean_fraction)))
    rng = random.Random(args.seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    sub_idx = idxs[:k]
    clean_subset = Subset(full_train, sub_idx)

    clean_loader = DataLoader(
        clean_subset,
        batch_size=max(64, args.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load frozen victim
    base = build_vit_model(model_name, pretrained=False, num_classes=num_cls)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    base.load_state_dict(ckpt, strict=True)
    base.to(dev)
    base.eval()

    after_block = args.fs_after_block if args.fs_after_block >= 0 else None

    model_name = getattr(args, 'model', 'deit_tiny')
    model_fs = build_fs_wrapper(
        base,
        model_name=model_name,
        use_layernorm_after_p=args.use_ln_after_fs,
        split_cls=args.fs_split_cls,
        after_block=after_block,
        token_drop_prob=getattr(args, 'token_drop_prob', 0.0)
    ).to(dev)

    model_fs.freeze_base()
    model_fs.train(False)  # keep base deterministic; token dropout is controlled by enable_token_dropout

    params = [p for p in model_fs.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr_fs)
    scheduler_fs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fs_epochs)
    grad_clip = getattr(args, 'grad_clip', 1.0)

    proxy_pool = build_proxy_attack_list(seed=args.seed)

    best_metric = float('inf')

    for epoch in range(1, args.fs_epochs + 1):
        epoch_loss = 0.0
        for x, y in tqdm(clean_loader, desc=f"[FS-UNIV] Epoch {epoch}/{args.fs_epochs}"):
            x = x.to(dev, non_blocking=True)  # normalized
            y = y.to(dev, non_blocking=True)


            # 1) Optional shape-agnostic degradations in pixel space (no explicit trigger templates)
            x_aug = x
            if getattr(args, 'use_common_degrade', False):
                x_pix = denorm(x_aug, mean, std).clamp(0.0, 1.0)
                x_pix = common_degrade_pixelspace(x_pix, prob=getattr(args, 'degrade_prob', 0.5))
                x_aug = renorm(x_pix, mean, std)

            # 2) Optionally stamp proxy trigger on batch (can be disabled by setting proxy_triggers_prob=0)
            if args.proxy_triggers_prob > 0.0:
                x_aug = maybe_apply_proxy_triggers(
                    x_norm_batch=x_aug,
                    mean=mean, std=std,
                    proxy_pool=proxy_pool,
                    prob=args.proxy_triggers_prob
                )

            # 3) Get baseline logits & pseudo-target for PGD direction
            with torch.no_grad():
                logits_clean_for_dir = forward_logits(model_fs, x_aug)
                y_hat = logits_clean_for_dir.argmax(dim=1)

            # 4) Sample eps and steps for broader coverage
            eps_list_str = getattr(args, 'pgd_eps_list', '')
            if isinstance(eps_list_str, str) and eps_list_str.strip():
                eps_candidates = [float(v.strip()) / 255.0 for v in eps_list_str.split(',') if v.strip()]
                eps = eps_candidates[random.randrange(len(eps_candidates))]
            else:
                eps = args.eps

            steps_min = getattr(args, 'pgd_steps_min', None)
            steps_max = getattr(args, 'pgd_steps_max', None)
            steps_min = int(args.steps if steps_min is None else steps_min)
            steps_max = int(args.steps if steps_max is None else steps_max)
            if steps_max < steps_min:
                steps_max = steps_min
            steps = random.randint(steps_min, steps_max)

            # Enable token dropout only during FS training
            model_fs.enable_token_dropout = (getattr(args, 'token_drop_prob', 0.0) > 0.0)

            # 5) Strong PGD from x_aug, toward y_hat (random start improves coverage)
            x_adv = targeted_pgd_pixelspace(
                model=model_fs,
                x_norm=x_aug,
                y_tgt=y_hat,
                mean=mean,
                std=std,
                eps=float(eps),
                alpha=args.alpha,
                steps=int(steps),
                random_start=getattr(args, 'pgd_random_start', False),
            )

            # 6) Logits under FS for clean_aug and adv
            logits_clean = forward_logits(model_fs, x_aug)
            logits_adv = forward_logits(model_fs, x_adv)

            # 7) Loss terms
            ce_clean = F.cross_entropy(logits_clean, y)
            ce_adv = F.cross_entropy(logits_adv, y)

            with torch.no_grad():
                p_clean = F.softmax(logits_clean, dim=1)
            p_adv_log = F.log_softmax(logits_adv, dim=1)
            consistency = F.kl_div(p_adv_log, p_clean, reduction="batchmean")

            p_adv = F.softmax(logits_adv, dim=1)
            entropy_adv = -(p_adv * torch.log(p_adv + 1e-8)).sum(dim=1).mean()
            entropy_loss = -entropy_adv

            reg = model_fs.fs_reg_identity()

            # w_clean = float(getattr(args, 'lambda_ce_clean', 1.0))
            # w_adv = float(getattr(args, 'lambda_ce_adv', args.lambda1))
            # w_reg = float(getattr(args, 'lambda_reg', args.lambda3))

            w_clean = float(getattr(args, "lambda_ce_clean", 1.0))

            w_adv_raw = getattr(args, "lambda_ce_adv", None)
            w_adv = float(args.lambda1 if w_adv_raw is None else w_adv_raw)

            w_reg_raw = getattr(args, "lambda_reg", None)
            w_reg = float(args.lambda3 if w_reg_raw is None else w_reg_raw)


            loss = (
                w_clean * ce_clean
                + w_adv * ce_adv
                + args.lambda_consistency * consistency
                + args.lambda_entropy * entropy_loss
                + w_reg * reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        scheduler_fs.step()
        avg_loss = epoch_loss / len(clean_subset)

        # Quick clean accuracy check for checkpoint selection
        with torch.no_grad():
            eval_iter = iter(clean_loader)
            x_eval, y_eval = next(eval_iter)
            x_eval, y_eval = x_eval.to(dev), y_eval.to(dev)
            logits_eval = forward_logits(model_fs, x_eval)
            clean_acc_epoch = (logits_eval.argmax(1) == y_eval).float().mean().item()

        # Combined metric: low loss + high clean acc (lower is better)
        metric = avg_loss - 0.5 * clean_acc_epoch
        print(f"[FS] Epoch {epoch}: avg_loss={avg_loss:.4f}  batch_clean_acc={clean_acc_epoch:.4f}  "
              f"metric={metric:.4f}  lr={scheduler_fs.get_last_lr()[0]:.6f}")

        if metric < best_metric:
            best_metric = metric
            pack = {
                'fs_state_dict': model_fs.state_dict(),
                'meta': {
                    'mean': mean, 'std': std,
                    'eps': args.eps, 'alpha': args.alpha, 'steps': args.steps,
                    'lambda1': args.lambda1,
                    'lambda_consistency': args.lambda_consistency,
                    'lambda_entropy': args.lambda_entropy,
                    'lambda3': args.lambda3,
                    'use_layernorm_after_p': args.use_ln_after_fs,
                    'fs_after_block': args.fs_after_block,
                    'fs_split_cls': args.fs_split_cls,
                    'proxy_triggers_prob': args.proxy_triggers_prob,
                }
            }
            torch.save(pack, args.save_fs)
            print(f"  >> Saved FilterShield to {args.save_fs} (avg_loss={best_metric:.4f})")

    print("Universal FilterShield training done.")


# -----------------------------
# Build test loader
# -----------------------------

def build_test_loader(data_root: str, batch_size: int = 256,
                      offline: bool = False, no_download: bool = False,
                      dataset: str = 'cifar10'):
    _, test_tf, _, _ = build_transforms(img_size=224, dataset=dataset)
    if offline:
        _set_offline_env()
        if dataset == 'cifar10':
            _assert_cifar10_present(data_root)
    download_data = (not offline) and (not no_download)
    test_ds = build_dataset(dataset, data_root, transform=test_tf,
                            train=False, download=download_data)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return test_loader


# -----------------------------
# Evaluate (with/without FS) + Optional Visualization
# -----------------------------

def evaluate_models(args):
    set_seed(args.seed)
    dev = device()
    dataset_name = getattr(args, 'dataset', 'cifar10')
    model_name = getattr(args, 'model', 'deit_tiny')
    num_cls = NUM_CLASSES.get(dataset_name, 10)

    test_loader = build_test_loader(args.data,
                                    batch_size=max(128, args.batch_size),
                                    offline=args.offline,
                                    no_download=args.no_download,
                                    dataset=dataset_name)

    mean, std = DATASET_STATS.get(dataset_name, DATASET_STATS['cifar10'])

    src_set = set(args.poison_source_classes)

    base = build_vit_model(model_name, pretrained=False,
                           num_classes=num_cls).to(dev)
    base.load_state_dict(torch.load(args.victim_ckpt, map_location='cpu', weights_only=True), strict=True)

    attack = build_attack(args.attack, args)

    clean_acc = evaluate_clean_acc(base, test_loader, dev)
    asr = evaluate_asr(base, test_loader, dev, target_label=args.target, attack_obj=attack, poison_source_classes=args.poison_source_classes)
    print(f"[Victim] Clean ACC: {clean_acc:.2f}%   ASR: {asr:.2f}% ({args.attack})")

    if args.fs_ckpt is not None and os.path.isfile(args.fs_ckpt):
        pack = torch.load(args.fs_ckpt, map_location='cpu', weights_only=True)
        _ab = pack['meta'].get('fs_after_block', -1)
        model_name = getattr(args, 'model', 'deit_tiny')
        model_fs = build_fs_wrapper(
            base,
            model_name=model_name,
            use_layernorm_after_p=pack['meta'].get('use_layernorm_after_p', False),
            split_cls=pack['meta'].get('fs_split_cls', False),
            after_block=(_ab if _ab >= 0 else None),
        ).to(dev)
        model_fs.load_state_dict(pack['fs_state_dict'], strict=False)

        clean_acc_fs = evaluate_clean_acc(model_fs, test_loader, dev)
        asr_fs = evaluate_asr(model_fs, test_loader, dev, target_label=args.target, attack_obj=attack, poison_source_classes=args.poison_source_classes)
        print(f"[FS]     Clean ACC: {clean_acc_fs:.2f}%   ASR: {asr_fs:.2f}% ({args.attack})")
    else:
        print("[FS] No FilterShield checkpoint provided; skipping FS evaluation.")

    # Visualization of triggers on victim model's input space
    if args.viz_triggers and args.viz_triggers > 0:
        outdir = args.viz_outdir or "./viz_triggers"
        os.makedirs(outdir, exist_ok=True)
        print(f"[Viz] Start saving {args.viz_triggers} triptychs to: {outdir}")

        saved = 0
        for x, y in test_loader:
            if saved >= args.viz_triggers:
                break
            x = x.to(dev)
            y = y.to(dev)

            mask = torch.zeros_like(y, dtype=torch.bool)
            for cls in src_set:
                mask |= (y == cls)
            if mask.sum() == 0:
                continue

            x_pixel = denorm(x, mean, std).clamp(0, 1)
            x_sel = x_pixel[mask]
            batch_to_save = min(x_sel.size(0), args.viz_triggers - saved)
            x_sel = x_sel[:batch_to_save]
            save_viz_triptychs(
                x_pixel_batch=x_sel,
                attack_obj=attack,
                mean=mean,
                std=std,
                outdir=outdir,
                prefix=f"{args.attack}_idx{saved:04d}"
            )
            saved += batch_to_save

        print(f"[Viz] Done. Saved {saved} images in classes {args.poison_source_classes} to {outdir}.")


# -----------------------------
# Main / CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DeiT-tiny + CIFAR-10: Backdoor Attacks + Universal FilterShield Defense + Visualization"
    )

    parser.add_argument('--task', type=str, required=True,
                        choices=['train_victim', 'train_fs', 'eval'])
    parser.add_argument('--data', type=str, default='./data', help='Dataset root directory')
    # Phase 2: dataset and model selection (backward compatible defaults)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'gtsrb', 'tiny_imagenet'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--model', type=str, default='deit_tiny',
                        choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_b16', 'swin_t'],
                        help='ViT architecture (default: deit_tiny)')
    parser.add_argument('--epochs', type=int, default=100, help='Victim training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Victim learning rate')
    parser.add_argument('--wd', type=float, default=0.05, help='Victim weight decay')
    parser.add_argument('--target', type=int, default=0, help='Target label for attack when poisoning')
    parser.add_argument('--poison_rate', type=float, default=0.1,
                        help='Poison fraction of source-class samples (effective rate = poison_rate * |src_class| / |total|)')
    parser.add_argument('--poison_source_classes', nargs='+', type=int, default=[3], help='List of source classes for poisoning (default: [0])')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet-pretrained DeiT weights for victim')
    parser.add_argument('--pretrained_ckpt', type=str, default='',
                        help='Local DeiT pretrained checkpoint (state_dict or dict with key "model" / "state_dict"). '
                             'If set, the code will load weights from this file instead of downloading.')
    parser.add_argument('--offline', action='store_true',
                        help='Run in offline mode (no external downloads). Requires CIFAR-10 already present under --data and uses --pretrained_ckpt for pretrained weights.')
    parser.add_argument('--no_download', action='store_true',
                        help='Do not attempt to download CIFAR-10 even when not offline (assume it already exists).')
    parser.add_argument('--save', type=str, default='victim.pth', help='Where to save victim model')
    parser.add_argument('--victim_save_tol', type=float, default=2.0,
                        help='Victim save tolerance (pp): save checkpoint with highest ASR '
                             'as long as clean_acc >= best_ever_clean_acc - tol. '
                             'Default 2.0 means allow up to 2%% clean acc drop.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Attack selection and params
    parser.add_argument('--attack', type=str, default='badnets',
                        choices=['badnets', 'wanet', 'blend', 'sig', 'trojvit', 'badvit', 'issba'],
                        help='Attack type (evaluation: badnets/wanet/blend/trojvit/badvit; sig kept for backwards compat)')

    # BadNets
    parser.add_argument('--trigger_size', type=int, default=4, help='BadNets square size (pixels on 224)')

    # WaNet (follows original ICLR 2021 paper parameters)
    parser.add_argument('--wanet_s', type=float, default=0.5,
                        help='WaNet warping strength s (original paper default: 0.5)')
    parser.add_argument('--wanet_grid', type=int, default=4,
                        help='WaNet coarse grid size k for displacement field')
    parser.add_argument('--wanet_grid_rescale', type=float, default=1.0,
                        help='WaNet grid rescale factor')
    parser.add_argument('--wanet_cross_ratio', type=float, default=2.0,
                        help='WaNet noise mode ratio: noise_samples = cross_ratio * num_poison')

    # Blend
    parser.add_argument('--blend_alpha', type=float, default=0.2, help='Blend alpha')
    parser.add_argument('--blend_pattern', type=str, default='noise',
                        choices=['noise', 'checker', 'stripes'],
                        help='Blend pattern type')

    # SIG
    parser.add_argument('--sig_amplitude', type=float, default=0.2, help='SIG amplitude')
    parser.add_argument('--sig_freq', type=int, default=8, help='SIG frequency (cycles across axis)')
    parser.add_argument('--sig_axis', type=str, default='x', choices=['x', 'y'], help='SIG axis (x or y)')
    parser.add_argument('--sig_phase', type=float, default=0.0, help='SIG phase (radians)')

    # TrojViT
    parser.add_argument('--trojvit_center_row', type=int, default=12,
                        help='TrojViT trigger center row in patch grid (0-based)')
    parser.add_argument('--trojvit_center_col', type=int, default=12,
                        help='TrojViT trigger center col in patch grid (0-based)')
    parser.add_argument('--trojvit_radius_patches', type=int, default=1,
                        help='TrojViT radius (in patches)')
    parser.add_argument('--trojvit_alpha', type=float, default=0.9,
                        help='TrojViT blend alpha in [0,1]')
    parser.add_argument('--trojvit_pattern', type=str, default='checker',
                        choices=['checker', 'stripes', 'noise'],
                        help='TrojViT pattern')

    # BadViT (visible & invisible)
    parser.add_argument('--badvit_mode', type=str, default='visible',
                        choices=['visible', 'invisible'],
                        help='BadViT mode')
    parser.add_argument('--badvit_shape', type=str, default='cross',
                        choices=['cross', 'x', 'ring'],
                        help='BadViT geometric shape at patch level')
    parser.add_argument('--badvit_alpha', type=float, default=0.8,
                        help='(visible) blend alpha in [0,1]')
    parser.add_argument('--badvit_patch_size', type=int, default=16,
                        help='Patch size (should match ViT patch size)')
    parser.add_argument('--badvit_thickness', type=int, default=1,
                        help='Thickness of shape in patches')
    parser.add_argument('--badvit_norm', type=str, default='linf',
                        choices=['linf', 'l2'],
                        help='(invisible) norm type')
    parser.add_argument('--badvit_eps', type=float, default=8/255,
                        help='(invisible) epsilon bound')

    # ISSBA
    parser.add_argument('--issba_encoder', type=str, default='issba_encoder.pth',
                        help='Path to pretrained ISSBA steganography encoder')
    parser.add_argument('--issba_eps', type=float, default=0.04,
                        help='ISSBA perturbation clamp')

    # FS args (universal mode)
    parser.add_argument('--victim_ckpt', type=str, default='victim.pth',
                        help='Victim checkpoint to load for FS training / eval')
    parser.add_argument('--save_fs', type=str, default='fs_universal.pth',
                        help='Where to save FS checkpoint')
    parser.add_argument('--clean_fraction', type=float, default=0.05,
                        help='Fraction of training set as clean subset for FS')
    parser.add_argument('--fs_epochs', type=int, default=15,
                        help='FS epochs')
    parser.add_argument('--lr_fs', type=float, default=1e-3,
                        help='FS learning rate')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Weight for CE(x_adv, y)')
    parser.add_argument('--lambda_consistency', type=float, default=1.0,
                        help='Weight for KL(logits_adv || logits_clean)')
    parser.add_argument('--lambda_entropy', type=float, default=0.5,
                        help='Weight for entropy regularizer on logits_adv')
    parser.add_argument('--lambda3', type=float, default=1e-4,
                        help='L2 regularization on (P - I)')
    parser.add_argument('--eps', type=float, default=8/255,
                        help='PGD epsilon (pixel space)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD step size (pixel space)')
    parser.add_argument('--steps', type=int, default=7,
                        help='PGD steps')
    parser.add_argument('--use_ln_after_fs', action='store_true',
                        help='Add LayerNorm after P during FS')
    parser.add_argument('--fs_after_block', type=int, default=-1,
                        help='Apply FS after this block index (0-based). -1 means after pos_drop')
    parser.add_argument('--fs_split_cls', action='store_true',
                        help='Use separate P for CLS vs patch tokens')
    parser.add_argument('--proxy_triggers_prob', type=float, default=0.5,
                        help='Probability of stamping a proxy trigger on a batch during FS training')

    parser.add_argument('--pgd_eps_list', type=str, default='',
                    help='Comma-separated PGD eps candidates in 1/255 units, e.g. "2,4,8". If empty, uses --eps')
    parser.add_argument('--pgd_steps_min', type=int, default=None,
                    help='Min PGD steps for FS training. If None, uses --steps')
    parser.add_argument('--pgd_steps_max', type=int, default=None,
                    help='Max PGD steps for FS training. If None, uses --steps')
    parser.add_argument('--pgd_random_start', action='store_true',
                    help='Use random start inside eps box for PGD during FS training')

    parser.add_argument('--token_drop_prob', type=float, default=0.0,
                    help='Randomly drop patch tokens after FS during FS training (shape-agnostic)')
    parser.add_argument('--use_common_degrade', action='store_true',
                    help='Apply mild, shape-agnostic degradations during FS training')
    parser.add_argument('--degrade_prob', type=float, default=0.5,
                    help='Probability of applying common degradations')

    parser.add_argument('--lambda_ce_clean', type=float, default=1.0,
                    help='Weight for CE(x_clean, y) during FS training')
    parser.add_argument('--lambda_ce_adv', type=float, default=None,
                    help='Weight for CE(x_adv, y) during FS training. If None, uses --lambda1')
    parser.add_argument('--lambda_reg', type=float, default=None,
                    help='Weight for ||P - I||^2 during FS training. If None, uses --lambda3')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping max norm for FS training. 0 to disable.')

    # Eval args
    parser.add_argument('--fs_ckpt', type=str, default=None,
                        help='FS checkpoint to evaluate with victim')

    # Visualization args
    parser.add_argument('--viz_triggers', type=int, default=0,
                        help='Export N triptychs (orig/attacked/diff) from test set; 0 to disable')
    parser.add_argument('--viz_outdir', type=str, default='./viz_triggers',
                        help='Directory to save visualization images')

    args = parser.parse_args()

    if args.offline:
        _set_offline_env()

    if args.task == 'train_victim':
        train_victim(args)
    elif args.task == 'train_fs':
        train_fs(args)
    elif args.task == 'eval':
        evaluate_models(args)


if __name__ == '__main__':
    main()
