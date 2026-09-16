from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from eemf.method.constants import (
    DIST_EPS,
    DIST_MIN_VAR,
    SCORE_NORM_CLIP,
    SCORE_NORM_EPS,
    SCORE_NORM_MIN_COUNT,
    SCORE_NORM_MODE,
    TEXT_DIST_EPS,
    TEXT_SCORE_WEIGHT,
)


def _feature_float(feat: torch.Tensor) -> torch.Tensor:
    return feat.detach().float()


def _feature_key(feat: torch.Tensor):
    x = feat.detach()
    storage = x.untyped_storage() if hasattr(x, "untyped_storage") else x.storage()
    return (int(x.data_ptr()), int(storage.data_ptr()), tuple(x.shape), str(x.device))


def make_score_norm_state() -> Dict[str, Dict[str, float]]:
    return {
        "visual": {"count": 0, "mean": 0.0, "m2": 0.0},
        "text": {"count": 0, "mean": 0.0, "m2": 0.0},
    }


def _running_std(entry: Dict[str, float]) -> Optional[float]:
    count = int(entry["count"])
    if count < 2:
        return None
    return (float(entry["m2"]) / float(count - 1)) ** 0.5


def _score_norm_ready(score_norm_state, modalities: Iterable[str]) -> bool:
    if SCORE_NORM_MODE != "running_zscore" or score_norm_state is None:
        return False

    for modality in modalities:
        entry = score_norm_state[modality]
        std = _running_std(entry)
        if int(entry["count"]) < SCORE_NORM_MIN_COUNT or std is None or std < SCORE_NORM_EPS:
            return False
    return True


def _score_for_joint(score_norm_state, modality: str, raw_score):
    if raw_score is None:
        return None
    if SCORE_NORM_MODE != "running_zscore" or score_norm_state is None:
        return float(raw_score)

    entry = score_norm_state[modality]
    std = _running_std(entry)
    if int(entry["count"]) < SCORE_NORM_MIN_COUNT or std is None or std < SCORE_NORM_EPS:
        return float(raw_score)

    score = (float(raw_score) - float(entry["mean"])) / (std + SCORE_NORM_EPS)
    if SCORE_NORM_CLIP > 0:
        score = max(min(score, SCORE_NORM_CLIP), -SCORE_NORM_CLIP)
    return float(score)


def _update_running_score(entry: Dict[str, float], value: float) -> None:
    value = float(value)
    count_old = int(entry["count"])
    count_new = count_old + 1
    if count_old == 0:
        entry["count"] = 1
        entry["mean"] = value
        entry["m2"] = 0.0
        return

    delta = value - float(entry["mean"])
    mean_new = float(entry["mean"]) + delta / float(count_new)
    delta2 = value - mean_new
    entry["count"] = count_new
    entry["mean"] = mean_new
    entry["m2"] = float(entry["m2"]) + delta * delta2


def _update_score_norm_state(score_norm_state, score: Dict[str, Optional[float]]) -> None:
    if SCORE_NORM_MODE != "running_zscore" or score_norm_state is None:
        return
    for modality in ("visual", "text"):
        value = score.get(modality)
        if value is not None:
            _update_running_score(score_norm_state[modality], float(value))


def make_visual_distribution_state():
    return {
        "classes": {},
        "shared": {"count": 0.0, "degrees": 0.0, "m2": None},
    }


def update_visual_distribution_weighted(visual_dist, pred, feat: torch.Tensor, strength: float = 1.0) -> bool:
    pred = int(pred)
    weight = float(strength)
    if weight <= 0:
        return False

    classes = visual_dist["classes"]
    if pred not in classes:
        classes[pred] = {
            "count": 0.0,
            "sample_count": 0,
            "mean": None,
            "m2": None,
            "seen": set(),
        }

    entry = classes[pred]
    key = _feature_key(feat)
    if key in entry["seen"]:
        return False

    x = _feature_float(feat)
    entry["seen"].add(key)
    count_old = float(entry["count"])
    if count_old <= 0:
        entry["count"] = weight
        entry["sample_count"] = 1
        entry["mean"] = x.clone()
        entry["m2"] = torch.zeros_like(x)
    else:
        count_new = count_old + weight
        mean_old = entry["mean"]
        delta = x - mean_old
        mean_new = mean_old + (weight / count_new) * delta
        delta2 = x - mean_new
        entry["m2"] = entry["m2"] + weight * delta * delta2
        entry["mean"] = mean_new
        entry["count"] = count_new
        entry["sample_count"] = int(entry["sample_count"]) + 1

    total_count = 0.0
    total_degrees = 0.0
    pooled_m2 = None
    for class_entry in classes.values():
        class_count = float(class_entry["count"])
        total_count += class_count
        degrees = max(class_count - 1.0, 0.0)
        total_degrees += degrees
        if degrees > 0:
            pooled_m2 = class_entry["m2"] if pooled_m2 is None else pooled_m2 + class_entry["m2"]

    visual_dist["shared"]["count"] = float(total_count)
    visual_dist["shared"]["degrees"] = float(total_degrees)
    visual_dist["shared"]["m2"] = pooled_m2
    return True


def visual_distribution_entry(visual_dist, pred):
    pred = int(pred)
    if pred not in visual_dist["classes"]:
        return None

    entry = visual_dist["classes"][pred]
    sample_count = int(entry.get("sample_count", 0))
    shared = visual_dist["shared"]
    degrees = float(shared["degrees"])
    if sample_count < 2 or degrees <= 0 or shared["m2"] is None:
        return None

    var = (shared["m2"] / max(degrees, 1e-12)).clamp_min(DIST_MIN_VAR)
    return {
        "count": sample_count,
        "mean": entry["mean"],
        "var": var,
    }


def text_distribution_entry(text_dist, pred, ref_feat: torch.Tensor):
    if text_dist is None:
        return None
    pred = int(pred)
    if pred not in text_dist:
        return None

    entry = text_dist[pred]
    dtype = torch.float32
    return {
        "count": int(entry["count"]),
        "mean": entry["mean"].to(device=ref_feat.device, dtype=dtype),
        "var": entry["var"].to(device=ref_feat.device, dtype=dtype),
    }


def distribution_score(entry, feat: torch.Tensor, eps: float):
    if entry is None or int(entry["count"]) < 2:
        return None
    x = _feature_float(feat).to(device=entry["mean"].device, dtype=entry["mean"].dtype)
    raw = torch.mean(((x - entry["mean"]) ** 2) / (entry["var"] + eps))
    return float((-raw).detach().cpu().item())


def joint_distribution_score(visual_dist, text_dist, pred, feat: torch.Tensor, score_norm_state=None):
    visual_score = distribution_score(visual_distribution_entry(visual_dist, pred), feat, DIST_EPS)
    if visual_score is None:
        return None

    text_score = distribution_score(text_distribution_entry(text_dist, pred, feat), feat, TEXT_DIST_EPS)
    if text_score is None:
        visual_for_joint = _score_for_joint(score_norm_state, "visual", visual_score) if _score_norm_ready(score_norm_state, ("visual",)) else float(visual_score)
        return {
            "joint": float(visual_for_joint),
            "visual": float(visual_score),
            "text": None,
        }

    if _score_norm_ready(score_norm_state, ("visual", "text")):
        visual_for_joint = _score_for_joint(score_norm_state, "visual", visual_score)
        text_for_joint = _score_for_joint(score_norm_state, "text", text_score)
    else:
        visual_for_joint = float(visual_score)
        text_for_joint = float(text_score)

    return {
        "joint": float(visual_for_joint + TEXT_SCORE_WEIGHT * text_for_joint),
        "visual": float(visual_score),
        "text": float(text_score),
    }


def joint_quality_for_candidate(visual_dist, text_dist, score_norm_state, pred, feat: torch.Tensor):
    score = joint_distribution_score(visual_dist, text_dist, pred, feat, score_norm_state)
    if score is None:
        return None
    _update_score_norm_state(score_norm_state, score)
    return float(score["joint"])


def update_visual_distribution_for_dual_cache(
    visual_dist,
    entropy_cache_before,
    energy_cache_before,
    pred,
    feat: torch.Tensor,
    entropy_cap: int,
    energy_cap: int,
    entropy_accepted: bool,
    energy_accepted: bool,
) -> bool:
    pred = int(pred)
    entropy_count_before = len(entropy_cache_before.get(pred, []))
    energy_count_before = len(energy_cache_before.get(pred, []))
    cold_start = entropy_count_before < entropy_cap or energy_count_before < energy_cap

    if cold_start:
        if entropy_accepted or energy_accepted:
            return update_visual_distribution_weighted(visual_dist, pred, feat, strength=1.0)
        return False

    if entropy_accepted and energy_accepted:
        return update_visual_distribution_weighted(visual_dist, pred, feat, strength=1.1)
    return False
