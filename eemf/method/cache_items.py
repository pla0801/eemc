from __future__ import annotations

from typing import List

import torch


def scalar_value(value) -> float:
    if value is None:
        return float("-inf")
    if torch.is_tensor(value):
        return float(value.detach().float().view(-1)[0].cpu().item())
    return float(value)


def make_entropy_item(pc_feats: torch.Tensor, entropy) -> List[object]:
    return [pc_feats, scalar_value(entropy)]


def make_energy_item(pc_feats: torch.Tensor, energy) -> List[object]:
    return [pc_feats, scalar_value(energy)]


def make_local_item(patch_centers: torch.Tensor, quality_score) -> List[object]:
    return [patch_centers, scalar_value(quality_score)]


def make_negative_item(pc_feats: torch.Tensor, entropy, prob_map: torch.Tensor) -> List[object]:
    return [pc_feats, scalar_value(entropy), prob_map]

