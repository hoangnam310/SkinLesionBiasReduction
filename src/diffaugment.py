"""
Differentiable Augmentation (DiffAugment) for data-efficient GAN training.

Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN
Training", NeurIPS 2020. https://arxiv.org/abs/2006.10738

The same augmentation is applied to real and fake images before they are
passed to the critic/discriminator. Because every op is differentiable,
gradients still propagate from the critic back through the generator.

Inputs are expected in the cGAN convention used elsewhere in this repo:
shape (B, 3, H, W), values roughly in [-1, 1].
"""

from typing import Optional

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: Optional[str] = "color,translation,cutout") -> torch.Tensor:
    """
    Apply DiffAugment transforms.

    Args:
        x: Image batch (B, C, H, W).
        policy: Comma-separated policies. Subset of {"color", "translation", "cutout"}.
                Pass None or "" to disable.
    """
    if not policy:
        return x
    for p in policy.split(","):
        p = p.strip()
        if not p:
            continue
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    delta = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + delta


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0
    return (x - x_mean) * scale + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * scale + x_mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    B, C, H, W = x.shape
    shift_x = int(H * ratio + 0.5)
    shift_y = int(W * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(B, 1, 1), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(B, 1, 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=x.device),
        torch.arange(H, dtype=torch.long, device=x.device),
        torch.arange(W, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, H + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, W + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    B, C, H, W = x.shape
    cutout_h = int(H * ratio + 0.5)
    cutout_w = int(W * ratio + 0.5)
    offset_x = torch.randint(0, H + (1 - cutout_h % 2), size=(B, 1, 1), device=x.device)
    offset_y = torch.randint(0, W + (1 - cutout_w % 2), size=(B, 1, 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=x.device),
        torch.arange(cutout_h, dtype=torch.long, device=x.device),
        torch.arange(cutout_w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_h // 2, 0, H - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_w // 2, 0, W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


if __name__ == "__main__":
    # Smoke test: shape preservation and differentiability.
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64, requires_grad=True)
    for policy in ["color", "translation", "cutout", "color,translation,cutout", ""]:
        out = diff_augment(x, policy=policy or None)
        assert out.shape == x.shape, f"shape mismatch for policy={policy!r}: {out.shape}"
        if policy:
            out.sum().backward(retain_graph=True)
        print(f"policy={policy!r:35s} -> out range [{out.min():.3f}, {out.max():.3f}]")
    print("DiffAugment smoke test OK")
