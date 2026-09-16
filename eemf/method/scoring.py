from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from eemf.method.constants import (
    FIXED_ALPHA_G,
    FIXED_ALPHA_L,
    FIXED_ALPHA_N,
)


def compute_energy(logits: torch.Tensor) -> float:
    energy = -torch.logsumexp(logits.detach().float(), dim=1)
    return float(energy.view(-1)[0].detach().cpu().item())


def normalized_entropy(entropy, num_classes: int) -> float:
    if torch.is_tensor(entropy):
        entropy_value = float(entropy.detach().float().view(-1)[0].cpu().item())
    else:
        entropy_value = float(entropy)
    return entropy_value / math.log2(int(num_classes))


def _empty_score_like(pc_feats: torch.Tensor, clip_weights: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (pc_feats.size(0), clip_weights.size(1)),
        device=pc_feats.device,
        dtype=pc_feats.dtype,
    )


@torch.no_grad()
def compute_cache_score(
    pc_feats: torch.Tensor,
    cache,
    beta: float,
    clip_weights: torch.Tensor,
    neg_mask_thresholds: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            cache_values.append(item[2] if neg_mask_thresholds else int(class_index))

    if not cache_keys:
        return _empty_score_like(pc_feats, clip_weights)

    keys = torch.cat(cache_keys, dim=0).to(device=pc_feats.device, dtype=pc_feats.dtype).t().contiguous()
    if neg_mask_thresholds:
        values = torch.cat(cache_values, dim=0).to(device=pc_feats.device, dtype=pc_feats.dtype)
        values = ((values > neg_mask_thresholds[0]) & (values < neg_mask_thresholds[1])).to(dtype=pc_feats.dtype)
    else:
        labels = torch.tensor(cache_values, device=pc_feats.device, dtype=torch.long)
        values = F.one_hot(labels, num_classes=clip_weights.size(1)).to(dtype=pc_feats.dtype)

    affinity = pc_feats @ keys
    return torch.exp(float(beta) * (affinity - 1.0)) @ values


@torch.no_grad()
def compute_local_cache_score(patch_centers: torch.Tensor, local_cache, beta: float, clip_weights: torch.Tensor) -> torch.Tensor:
    cache_keys = []
    labels = []
    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            centers = item[0]
            cache_keys.append(centers)
            labels.extend([int(class_index)] * int(centers.shape[0]))

    if not cache_keys:
        return torch.zeros(
            (1, clip_weights.size(1)),
            device=patch_centers.device,
            dtype=patch_centers.dtype,
        )

    keys = torch.cat(cache_keys, dim=0).to(device=patch_centers.device, dtype=patch_centers.dtype).t().contiguous()
    value_labels = torch.tensor(labels, device=patch_centers.device, dtype=torch.long)
    values = F.one_hot(value_labels, num_classes=clip_weights.size(1)).to(dtype=patch_centers.dtype)
    query = patch_centers.mean(dim=0, keepdim=True)
    affinity = query @ keys
    return torch.exp(float(beta) * (affinity - 1.0)) @ values


def compose_logits(y_zero: torch.Tensor, y_entropy: torch.Tensor, y_local: torch.Tensor, y_negative: torch.Tensor) -> torch.Tensor:
    return y_zero + FIXED_ALPHA_G * y_entropy + FIXED_ALPHA_L * y_local - FIXED_ALPHA_N * y_negative
