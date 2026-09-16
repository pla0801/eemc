from __future__ import annotations

import numpy as np
import torch


def _device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(getattr(args, "device", 0)))
    return torch.device("cpu")


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs: torch.Tensor) -> torch.Tensor:
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def _move_feature(feat, args) -> torch.Tensor:
    device = _device(args)
    if isinstance(feat, list):
        feat = torch.cat(feat, dim=0)
    return feat.to(device=device, non_blocking=True)


@torch.no_grad()
def get_uni3d_logits(cache_type, feat, lm3d_model, clip_weights, args):
    feat = _move_feature(feat, args)

    if cache_type == "global":
        pc_feats = lm3d_model.encode_pc(feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights
    elif cache_type == "local":
        patch_centers = lm3d_model.encode_pc(feat)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * patch_centers.mean(0, keepdim=True) @ clip_weights
    else:
        pc_feats, patch_centers = lm3d_model.encode_pc(feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights

    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0].item())

    if cache_type == "global":
        return pc_feats, clip_logits, loss, prob_map, pred
    if cache_type == "local":
        return patch_centers, clip_logits, loss, prob_map, pred
    return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


@torch.no_grad()
def get_openshape_logits(cache_type, feat, lm3d_model, clip_weights, args):
    feat = _move_feature(feat, args)
    xyz = feat[:, :, :3]

    if cache_type == "global":
        pc_feats = lm3d_model(xyz, feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights
    elif cache_type == "local":
        patch_centers = lm3d_model(xyz, feat)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * patch_centers.mean(0, keepdim=True) @ clip_weights
    else:
        pc_feats, patch_centers = lm3d_model(xyz, feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights

    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0].item())

    if cache_type == "global":
        return pc_feats, clip_logits, loss, prob_map, pred
    if cache_type == "local":
        return patch_centers, clip_logits, loss, prob_map, pred
    return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


@torch.no_grad()
def get_ulip_logits(cache_type, feat, lm3d_model, clip_weights, args):
    feat = _move_feature(feat, args)
    xyz = feat[:, :, :3]

    if cache_type == "global":
        pc_feats = lm3d_model(xyz)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights
    elif cache_type == "local":
        patch_centers = lm3d_model(xyz)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * patch_centers.mean(0, keepdim=True) @ clip_weights
    else:
        pc_feats, patch_centers = lm3d_model(xyz)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_logits = 100.0 * pc_feats @ clip_weights

    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0].item())

    if cache_type == "global":
        return pc_feats, clip_logits, loss, prob_map, pred
    if cache_type == "local":
        return patch_centers, clip_logits, loss, prob_map, pred
    return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


def get_logits(args, feat, lm3d_model, clip_weights):
    if args.lm3d == "uni3d":
        return get_uni3d_logits(args.cache_type, feat, lm3d_model, clip_weights, args)
    if args.lm3d == "openshape":
        return get_openshape_logits(args.cache_type, feat, lm3d_model, clip_weights, args)
    if args.lm3d == "ulip":
        return get_ulip_logits(args.cache_type, feat, lm3d_model, clip_weights, args)
    raise NotImplementedError(f"Unsupported 3D backbone: {args.lm3d}")
