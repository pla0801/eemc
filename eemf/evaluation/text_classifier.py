from __future__ import annotations

import torch


def _device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(getattr(args, "device", 0)))
    return torch.device("cpu")


def _lookup_class_prompts(classname, prompt_dict):
    raw_name = str(classname)
    clean_name = raw_name.replace("_", " ")
    candidate_keys = [
        raw_name,
        clean_name,
        raw_name.lower(),
        clean_name.lower(),
        raw_name.replace("_", " ").lower(),
    ]
    for key in candidate_keys:
        if key in prompt_dict:
            return prompt_dict[key]
    raise KeyError(f"No prompts found for class '{classname}'.")


def build_prompt_texts(classname, template):
    clean_name = str(classname).replace("_", " ")
    if isinstance(template, list):
        return [t.format(clean_name) for t in template]
    if isinstance(template, tuple):
        return [t.format(clean_name) for t in template]
    if isinstance(template, dict):
        return _lookup_class_prompts(str(classname), template)
    raise TypeError(f"Unsupported prompt template type: {type(template)}")


def is_weighted_prompt_fusion(template) -> bool:
    return (
        isinstance(template, dict)
        and template.get("__eemf_prompt_type__") == "weighted_fusion"
    )


def _tokenize_with_clip(texts, args):
    import clip

    return clip.tokenize(texts).to(_device(args), non_blocking=True)


@torch.no_grad()
def encode_texts_as_prototype(texts, clip_model, args):
    tokenized_texts = _tokenize_with_clip(texts, args)
    class_embeddings = clip_model.encode_text(tokenized_texts)
    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding = class_embedding / class_embedding.norm().clamp_min(1e-12)
    return class_embedding


@torch.no_grad()
def encode_weighted_prompt_fusion(classname, template, clip_model, args):
    static_template = template["static_template"]
    dynamic_template = template["dynamic_template"]
    static_weight = float(template.get("static_weight", 0.75))
    dynamic_weight = float(template.get("dynamic_weight", 0.25))

    static_texts = build_prompt_texts(classname, static_template)
    dynamic_texts = build_prompt_texts(classname, dynamic_template)
    static_embedding = encode_texts_as_prototype(static_texts, clip_model, args)
    dynamic_embedding = encode_texts_as_prototype(dynamic_texts, clip_model, args)

    class_embedding = static_weight * static_embedding + dynamic_weight * dynamic_embedding
    class_embedding = class_embedding / class_embedding.norm().clamp_min(1e-12)
    return class_embedding


@torch.no_grad()
def clip_classifier(args, classnames, template, clip_model):
    clip_weights = []
    for classname in classnames:
        if is_weighted_prompt_fusion(template):
            class_embedding = encode_weighted_prompt_fusion(classname, template, clip_model, args)
        else:
            texts = build_prompt_texts(classname, template)
            class_embedding = encode_texts_as_prototype(texts, clip_model, args)
        clip_weights.append(class_embedding)
    return torch.stack(clip_weights, dim=1).to(_device(args), non_blocking=True)
