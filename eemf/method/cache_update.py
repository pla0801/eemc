from __future__ import annotations

from typing import Dict, Optional

from eemf.method.cache_items import scalar_value


def _inc(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + 1


def _score(item, index: int) -> float:
    if not isinstance(item, (list, tuple)) or len(item) <= index:
        raise ValueError("Cache item missing score at index {}".format(index))
    return scalar_value(item[index])


def _sort_cache(cache, pred: int, score_index: int, reverse: bool = False) -> None:
    cache[pred] = sorted(cache[pred], key=lambda item: _score(item, score_index), reverse=reverse)


def _update_lower_is_better_cache(cache, pred, item, capacity: int, stats, phase: str, name: str) -> bool:
    pred = int(pred)
    if capacity <= 0:
        _inc(stats, "{}_{}_reject_capacity_zero".format(phase, name))
        return False

    if pred not in cache:
        cache[pred] = [item]
        _inc(stats, "{}_{}_add".format(phase, name))
        return True

    if len(cache[pred]) < capacity:
        cache[pred].append(item)
        _sort_cache(cache, pred, 1)
        _inc(stats, "{}_{}_add".format(phase, name))
        return True

    worst_index = max(range(len(cache[pred])), key=lambda idx: _score(cache[pred][idx], 1))
    worst_score = _score(cache[pred][worst_index], 1)
    curr_score = _score(item, 1)
    if curr_score < worst_score:
        cache[pred][worst_index] = item
        _sort_cache(cache, pred, 1)
        _inc(stats, "{}_{}_replace".format(phase, name))
        return True

    _inc(stats, "{}_{}_reject".format(phase, name))
    return False


def update_entropy_cache(cache, pred, item, capacity: int, stats=None, phase: str = "test") -> bool:
    return _update_lower_is_better_cache(cache, pred, item, capacity, stats, phase, "entropy")


def update_energy_cache(cache, pred, item, capacity: int, stats=None, phase: str = "test") -> bool:
    return _update_lower_is_better_cache(cache, pred, item, capacity, stats, phase, "energy")


def update_negative_cache(cache, pred, item, capacity: int, stats=None, phase: str = "test") -> bool:
    return _update_lower_is_better_cache(cache, pred, item, capacity, stats, phase, "negative")


def _local_quality(item) -> float:
    return _score(item, 1)


def _sort_local_cache(local_cache, pred: int) -> None:
    local_cache[pred] = sorted(local_cache[pred], key=_local_quality, reverse=True)


def update_dual_local_cache(
    local_cache,
    pred,
    item,
    capacity: int,
    entropy_accepted: bool,
    energy_accepted: bool,
    stats=None,
    phase: str = "test",
):
    pred = int(pred)
    if capacity <= 0:
        _inc(stats, "{}_local_reject_capacity_zero".format(phase))
        return {"accepted": False, "reason": "capacity_zero"}

    if pred not in local_cache:
        local_cache[pred] = []

    if entropy_accepted and energy_accepted:
        reason = "both"
    elif entropy_accepted:
        reason = "entropy_only"
    elif energy_accepted:
        reason = "energy_only"
    else:
        reason = "none"

    if len(local_cache[pred]) < capacity:
        if reason == "none":
            _inc(stats, "{}_local_reject_no_positive_cache".format(phase))
            return {"accepted": False, "reason": "positive_cache_reject"}

        local_cache[pred].append(item)
        _sort_local_cache(local_cache, pred)
        _inc(stats, "{}_local_add".format(phase))
        _inc(stats, "{}_local_accept_count".format(phase))
        _inc(stats, "{}_local_accept_by_{}".format(phase, reason))
        return {"accepted": True, "reason": "cold_start_{}".format(reason)}

    if not (entropy_accepted and energy_accepted):
        _inc(stats, "{}_local_reject_not_both".format(phase))
        return {"accepted": False, "reason": "mature_not_both"}

    curr_quality = _local_quality(item)
    if curr_quality == float("-inf"):
        _inc(stats, "{}_local_reject_no_distribution".format(phase))
        return {"accepted": False, "reason": "mature_no_distribution"}

    worst_index = min(range(len(local_cache[pred])), key=lambda idx: _local_quality(local_cache[pred][idx]))
    worst_quality = _local_quality(local_cache[pred][worst_index])
    if curr_quality > worst_quality:
        local_cache[pred][worst_index] = item
        _sort_local_cache(local_cache, pred)
        _inc(stats, "{}_local_replace_distribution".format(phase))
        _inc(stats, "{}_local_accept_count".format(phase))
        _inc(stats, "{}_local_accept_by_both".format(phase))
        return {
            "accepted": True,
            "reason": "mature_distribution_accept",
            "old_quality": worst_quality,
            "new_quality": curr_quality,
        }

    _inc(stats, "{}_local_reject_distribution".format(phase))
    return {
        "accepted": False,
        "reason": "mature_distribution_reject",
        "old_quality": worst_quality,
        "new_quality": curr_quality,
    }
