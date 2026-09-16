from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from eemf.method.cache_items import (
    make_energy_item,
    make_entropy_item,
    make_local_item,
    make_negative_item,
)
from eemf.method.cache_update import (
    update_dual_local_cache,
    update_energy_cache,
    update_entropy_cache,
    update_negative_cache,
)
from eemf.method.constants import (
    ENERGY_CAP,
    ENTROPY_CAP,
    LOCAL_CAP,
    NEG_CAP,
)
from eemf.method.distributions import (
    joint_quality_for_candidate,
    make_score_norm_state,
    make_visual_distribution_state,
    update_visual_distribution_for_dual_cache,
)
from eemf.method.scoring import (
    compose_logits,
    compute_cache_score,
    compute_energy,
    compute_local_cache_score,
    normalized_entropy,
)


@dataclass(frozen=True)
class StreamResult:
    acc: float
    total: int


def _feature_from_batch(pc, rgb):
    return torch.cat([pc, rgb], dim=-1).half()


def _target_on_device(target, ref: torch.Tensor) -> torch.Tensor:
    return target.to(device=ref.device, non_blocking=True).view(-1)


def _get_logits(args, feature, lm3d_model, clip_weights):
    from eemf.evaluation.backbone_logits import get_logits

    return get_logits(args, feature, lm3d_model, clip_weights)


def _update_positive_state(
    entropy_cache,
    energy_cache,
    local_cache,
    visual_dist,
    text_dist,
    score_norm_state,
    pred,
    pc_feats,
    patch_centers,
    entropy_value,
    energy_value,
):
    pred_int = int(pred)
    entropy_before = {pred_int: list(entropy_cache.get(pred_int, []))}
    energy_before = {pred_int: list(energy_cache.get(pred_int, []))}
    quality_score = joint_quality_for_candidate(
        visual_dist,
        text_dist,
        score_norm_state,
        pred_int,
        pc_feats,
    )

    entropy_accepted = update_entropy_cache(
        entropy_cache,
        pred_int,
        make_entropy_item(pc_feats, entropy_value),
        ENTROPY_CAP,
    )
    energy_accepted = update_energy_cache(
        energy_cache,
        pred_int,
        make_energy_item(pc_feats, energy_value),
        ENERGY_CAP,
    )
    update_dual_local_cache(
        local_cache,
        pred_int,
        make_local_item(patch_centers, quality_score),
        LOCAL_CAP,
        entropy_accepted,
        energy_accepted,
    )
    update_visual_distribution_for_dual_cache(
        visual_dist,
        entropy_before,
        energy_before,
        pred_int,
        pc_feats,
        ENTROPY_CAP,
        ENERGY_CAP,
        entropy_accepted,
        energy_accepted,
    )


@torch.no_grad()
def build_positive_caches(args, test_loader, lm3d_model, clip_weights, text_dist=None):
    entropy_cache = {}
    energy_cache = {}
    local_cache = {}
    visual_dist = make_visual_distribution_state()
    score_norm_state = make_score_norm_state()

    for pc, _target, _meta, rgb in test_loader:
        feature = _feature_from_batch(pc, rgb)
        pc_feats, patch_centers, clip_logits, entropy_value, _prob_map, pred = _get_logits(
            args,
            feature,
            lm3d_model,
            clip_weights,
        )
        _update_positive_state(
            entropy_cache,
            energy_cache,
            local_cache,
            visual_dist,
            text_dist,
            score_norm_state,
            pred,
            pc_feats,
            patch_centers,
            entropy_value,
            compute_energy(clip_logits),
        )

        num_classes = int(clip_logits.size(1))
        if (
            sum(len(v) for v in entropy_cache.values()) >= ENTROPY_CAP * num_classes
            and sum(len(v) for v in energy_cache.values()) >= ENERGY_CAP * num_classes
            and sum(len(v) for v in local_cache.values()) >= LOCAL_CAP * num_classes
        ):
            break

    return entropy_cache, energy_cache, local_cache, visual_dist, score_norm_state


@torch.no_grad()
def run_stream(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None) -> StreamResult:
    print("---- building positive caches .... ----", flush=True)
    entropy_cache, energy_cache, local_cache, visual_dist, score_norm_state = build_positive_caches(
        args,
        test_loader,
        lm3d_model,
        clip_weights,
        text_dist=text_dist,
    )
    print("---- positive caches ready; evaluating test stream ! -----", flush=True)

    pos_enabled = bool(pos_cfg.get("enabled", True))
    neg_enabled = bool(neg_cfg.get("enabled", True))
    pos_beta = float(pos_cfg.get("beta", 1.0))
    neg_beta = float(neg_cfg.get("beta", 1.0))
    neg_entropy_threshold = neg_cfg.get("entropy_threshold", {"lower": 0.0, "upper": 1.0})
    neg_mask_threshold = neg_cfg.get("mask_threshold", {"lower": 0.0, "upper": 1.0})

    negative_cache = {}
    correct = 0
    total = 0
    print_freq = max(1, int(getattr(args, "print_freq", 500)))

    for batch_idx, (pc, target, _meta, rgb) in enumerate(test_loader):
        feature = _feature_from_batch(pc, rgb)
        pc_feats, patch_centers, clip_logits, entropy_value, prob_map, pred = _get_logits(
            args,
            feature,
            lm3d_model,
            clip_weights,
        )

        if pos_enabled:
            _update_positive_state(
                entropy_cache,
                energy_cache,
                local_cache,
                visual_dist,
                text_dist,
                score_norm_state,
                pred,
                pc_feats,
                patch_centers,
                entropy_value,
                compute_energy(clip_logits),
            )

        prop_entropy = normalized_entropy(entropy_value, clip_weights.size(1))
        if (
            neg_enabled
            and float(neg_entropy_threshold["lower"]) < prop_entropy < float(neg_entropy_threshold["upper"])
        ):
            update_negative_cache(
                negative_cache,
                pred,
                make_negative_item(pc_feats, entropy_value, prob_map),
                NEG_CAP,
            )

        y_zero = clip_logits
        y_entropy = torch.zeros_like(y_zero)
        y_local = torch.zeros_like(y_zero)
        y_negative = torch.zeros_like(y_zero)

        if pos_enabled:
            if entropy_cache:
                y_entropy = compute_cache_score(pc_feats, entropy_cache, pos_beta, clip_weights)
            if local_cache:
                y_local = compute_local_cache_score(patch_centers, local_cache, pos_beta, clip_weights)

        if neg_enabled and negative_cache:
            y_negative = compute_cache_score(
                pc_feats,
                negative_cache,
                neg_beta,
                clip_weights,
                (float(neg_mask_threshold["lower"]), float(neg_mask_threshold["upper"])),
            )

        final_logits = compose_logits(y_zero, y_entropy, y_local, y_negative)
        pred_final = final_logits.argmax(dim=1)
        target_device = _target_on_device(target, final_logits)
        correct += int((pred_final == target_device).sum().detach().cpu().item())
        total += int(target_device.numel())

        if batch_idx % print_freq == 0:
            running_acc = 100.0 * float(correct) / float(total)
            print(
                f">>> tested samples: {total}, cumulative accuracy: {running_acc:.2f} <<<",
                flush=True,
            )

    if total <= 0:
        raise ValueError("No samples were evaluated")
    final_acc = 100.0 * float(correct) / float(total)
    print(
        f">>> final tested samples: {total}, cumulative accuracy: {final_acc:.2f} <<<",
        flush=True,
    )
    return StreamResult(acc=final_acc, total=total)
