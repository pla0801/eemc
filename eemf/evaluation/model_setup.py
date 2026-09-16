from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Dict, Iterable

import torch
import yaml

from data_utils.prompt_utils import get_prompt_template
from eemf.evaluation.dataloaders import build_test_data_loader
from eemf.evaluation.model_loader import load_models
from eemf.evaluation.text_classifier import (
    build_prompt_texts,
    clip_classifier,
    is_weighted_prompt_fusion,
)
from eemf.evaluation.protocols import apply_dataset_args
from eemf.method.constants import (
    ENTROPY_CAP,
    LOCAL_CENTERS,
    NEG_CAP,
    TEXT_DISTRIBUTION_PROMPT_SOURCE,
    TEXT_DIST_MIN_VAR,
    ZERO_SHOT_PROMPT_SOURCE,
)
from eemf.utils.seed import set_random_seed
from llm.prompt_generator import get_prompt_cache_path


def _device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(getattr(args, "device", 0)))
    return torch.device("cpu")


def get_backbone_name(args) -> str:
    if args.lm3d == "ulip":
        return "ULIP-2" if getattr(args, "ulip_version", "") == "ulip2" else "ULIP"
    if args.lm3d == "openshape":
        return "OpenShape"
    if args.lm3d == "uni3d":
        return "Uni3D"
    return str(args.lm3d)


def _first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return path
    return paths[0]


def apply_runtime_defaults(args, dataset_spec, backbone_spec) -> None:
    apply_dataset_args(args, dataset_spec)
    args.cache_type = "hierarchical"
    args.prompt_source = ZERO_SHOT_PROMPT_SOURCE
    args.k_shot = int(ENTROPY_CAP)
    args.n_cluster = int(LOCAL_CENTERS)
    args.text_dist_prompt_source = TEXT_DISTRIBUTION_PROMPT_SOURCE

    if backbone_spec.key == "ulip":
        args.lm3d = "ulip"
        args.ulip_version = "ulip1"
        args.ckpt_path = args.ckpt_path or "weights/ulip/pointbert_ulip1.pt"
        args.slip_ckpt_path = args.slip_ckpt_path or "weights/ulip/slip_base_100ep.pt"
        args.oshape_version = "vitg14"
    elif backbone_spec.key == "ulip2":
        args.lm3d = "ulip"
        args.ulip_version = "ulip2"
        args.ckpt_path = args.ckpt_path or "weights/ulip/pointbert_ulip2.pt"
        args.slip_ckpt_path = args.slip_ckpt_path or "weights/ulip/slip_base_100ep.pt"
        args.oshape_version = "vitg14"
    elif backbone_spec.key == "openshape":
        args.lm3d = "openshape"
        args.ckpt_path = args.ckpt_path or f"weights/openshape/openshape-pointbert-{args.oshape_version}-rgb/model.pt"
        args.ulip_version = args.ulip_version or "ulip2"
    elif backbone_spec.key == "uni3d":
        args.lm3d = "uni3d"
        args.pc_feat_dim = 1408
        args.num_group = 512
        args.group_size = 64
        args.pc_encoder_dim = 512
        args.embed_dim = 1024
        if not args.ckpt_path:
            if dataset_spec.dataset_key in {"scanobjnn", "scanobjnn_c"}:
                args.ckpt_path = "weights/uni3d/scanobjnn/model.pt"
            else:
                args.ckpt_path = "weights/uni3d/modelnet40/model.pt"
        args.pretrained = args.pretrained or _first_existing([
            "weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin",
            "weights/uni3d/open_clip_pytorch_model/eva02_enormous_patch14_plus_clip_224/laion2b_s9b_b144k.bin",
            "weights/uni3d/laion2b_s9b_b144k.bin",
        ])
        args.oshape_version = "vitg14"
        args.ulip_version = args.ulip_version or "ulip2"
    else:
        raise ValueError(f"Unsupported backbone: {backbone_spec.key}")


def backbone_matches(backbone_spec, actual_backbone: str) -> bool:
    if backbone_spec.key == "openshape":
        return str(actual_backbone).startswith("OpenShape")
    return str(actual_backbone) == backbone_spec.display_name


def build_clip_weights_once(args, clip_model, state: Dict[str, object], classnames, template):
    classnames = list(classnames)
    if state.get("clip_weights") is None:
        state["classnames"] = classnames
        state["template"] = template
        state["clip_weights"] = clip_classifier(args, classnames, template, clip_model)
        return state["clip_weights"]

    if classnames != state.get("classnames"):
        raise RuntimeError("Class names changed inside one run; rebuild text prototypes with a separate run.")
    return state["clip_weights"]


def _tokenize_texts(args, texts: Iterable[str]) -> torch.Tensor:
    texts = list(texts)
    if args.lm3d in {"uni3d", "ulip"}:
        import clip

        return clip.tokenize(texts).to(_device(args), non_blocking=True)
    if args.lm3d == "openshape":
        import open_clip

        return open_clip.tokenizer.tokenize(texts).to(_device(args), non_blocking=True)
    raise ValueError(f"Unsupported 3D backbone: {args.lm3d}")


@torch.no_grad()
def _encode_prompt_embeddings(args, clip_model, texts: Iterable[str]) -> torch.Tensor:
    tokenized = _tokenize_texts(args, texts)
    embeddings = clip_model.encode_text(tokenized)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return embeddings.float()


def _distribution_from_prompt_embeddings(class_embeddings: torch.Tensor, min_var: float):
    mean = class_embeddings.mean(dim=0, keepdim=True)
    if class_embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * float(min_var)
    else:
        var = class_embeddings.var(dim=0, unbiased=True, keepdim=True).clamp_min(float(min_var))
    return {
        "count": int(class_embeddings.size(0)),
        "mean": mean.detach(),
        "var": var.detach(),
    }


def _distribution_from_weighted_prompt_embeddings(class_embeddings: torch.Tensor, weights: torch.Tensor, min_var: float):
    weights = weights.to(device=class_embeddings.device, dtype=class_embeddings.dtype).view(-1, 1)
    weights = weights / weights.sum().clamp_min(1e-12)
    mean = (class_embeddings * weights).sum(dim=0, keepdim=True)
    if class_embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * float(min_var)
    else:
        var = ((class_embeddings - mean).pow(2) * weights).sum(dim=0, keepdim=True).clamp_min(float(min_var))
    return {
        "count": int(class_embeddings.size(0)),
        "mean": mean.detach(),
        "var": var.detach(),
    }


def _apply_shared_text_variance(text_dist: Dict[int, dict], min_var: float):
    if not text_dist:
        raise ValueError("Cannot build text distribution from empty class list")
    variances = [entry["var"].float() for entry in text_dist.values()]
    shared_variance = torch.stack(variances, dim=0).mean(dim=0).clamp_min(float(min_var)).detach()
    for entry in text_dist.values():
        entry["var"] = shared_variance
    return text_dist


def _build_weighted_fusion_prompt_distribution(args, clip_model, classname: str, template, min_var: float):
    static_texts = build_prompt_texts(classname, template["static_template"])
    dynamic_texts = build_prompt_texts(classname, template["dynamic_template"])
    static_embeddings = _encode_prompt_embeddings(args, clip_model, static_texts)
    dynamic_embeddings = _encode_prompt_embeddings(args, clip_model, dynamic_texts)
    class_embeddings = torch.cat([static_embeddings, dynamic_embeddings], dim=0)

    static_weight = float(template.get("static_weight", 0.75))
    dynamic_weight = float(template.get("dynamic_weight", 0.25))
    static_weights = torch.full(
        (static_embeddings.size(0),),
        static_weight / float(max(static_embeddings.size(0), 1)),
        device=class_embeddings.device,
        dtype=class_embeddings.dtype,
    )
    dynamic_weights = torch.full(
        (dynamic_embeddings.size(0),),
        dynamic_weight / float(max(dynamic_embeddings.size(0), 1)),
        device=class_embeddings.device,
        dtype=class_embeddings.dtype,
    )
    return _distribution_from_weighted_prompt_embeddings(
        class_embeddings,
        torch.cat([static_weights, dynamic_weights], dim=0),
        min_var,
    )


@torch.no_grad()
def build_text_distribution_once(args, clip_model, state: Dict[str, object], classnames, template):
    classnames = list(classnames)
    cached = state.get("text_dist")
    if cached is not None:
        if classnames != state.get("classnames"):
            raise RuntimeError("Class names changed inside one run; rebuild text distributions with a separate run.")
        return cached

    min_var = float(TEXT_DIST_MIN_VAR)
    text_dist: Dict[int, dict] = {}
    for class_index, classname in enumerate(classnames):
        if is_weighted_prompt_fusion(template):
            text_dist[int(class_index)] = _build_weighted_fusion_prompt_distribution(
                args,
                clip_model,
                classname,
                template,
                min_var,
            )
        else:
            texts = build_prompt_texts(classname, template)
            embeddings = _encode_prompt_embeddings(args, clip_model, texts)
            text_dist[int(class_index)] = _distribution_from_prompt_embeddings(embeddings, min_var)

    state["classnames"] = classnames
    state["text_dist"] = _apply_shared_text_variance(text_dist, min_var)
    return state["text_dist"]


def build_text_distribution_template(args, classnames, fallback_template):
    source = getattr(args, "text_dist_prompt_source", TEXT_DISTRIBUTION_PROMPT_SOURCE)
    if source == getattr(args, "prompt_source", ""):
        return fallback_template

    text_args = copy(args)
    text_args.prompt_source = source
    return get_prompt_template(text_args, classnames, dataset_name=args.dataset)


def validate_text_prompt_cache(args, prompt_dataset: str, project_root: Path) -> Path:
    source = getattr(args, "text_dist_prompt_source", TEXT_DISTRIBUTION_PROMPT_SOURCE)
    if source not in {"llm_dynamic_init", "manualfull_llm_dynamic_init"}:
        return Path("")

    class_file = project_root / "data" / prompt_dataset / "shape_names.txt"
    if not class_file.exists():
        raise FileNotFoundError(f"Missing class file: {class_file.relative_to(project_root)}")

    cache_path = get_prompt_cache_path(args, prompt_dataset)
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text prompt cache: {cache_path.relative_to(project_root)}")

    with class_file.open("r", encoding="utf-8") as f:
        classnames = [line.strip().replace("_", " ") for line in f if line.strip()]
    with cache_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    prompts = saved.get("prompts", saved)
    if not isinstance(prompts, dict):
        raise ValueError(f"Invalid text prompt cache format: {cache_path.relative_to(project_root)}")

    required_count = int(getattr(args, "dynamic_prompt_count", 10))
    missing = []
    short = []
    for classname in classnames:
        class_prompts = prompts.get(classname)
        if class_prompts is None:
            missing.append(classname)
        elif len(class_prompts) < required_count:
            short.append((classname, len(class_prompts)))

    if missing or short:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if short:
            details.append("short=" + ",".join(f"{name}:{count}" for name, count in short))
        raise ValueError(f"Incomplete text prompt cache {cache_path.relative_to(project_root)}: {'; '.join(details)}")

    return cache_path


def build_release_loader(args):
    return build_test_data_loader(args)


def load_release_models(args):
    return load_models(args)


def load_release_config(args):
    config_file = Path(args.config_dir) / f"{args.dataset}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config file: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["positive"]["shot_capacity"] = int(ENTROPY_CAP)
    cfg["positive"]["beta"] = float(cfg["positive"].get("beta", 3.0))
    cfg["negative"]["shot_capacity"] = int(NEG_CAP)
    return cfg


def set_release_seed(seed: int) -> None:
    set_random_seed(seed)
