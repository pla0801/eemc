"""Prompt-source utilities."""

from .templates import text_prompts, manual_3d_prompts
from llm.prompt_generator import generate_llm_prompts


def get_prompt_template(cfg, classnames=None, dataset_name=None):
    """Return prompt template according to cfg.prompt_source.

    Prompt sources can be a shared manual template list, an LLM prompt cache,
    or a weighted fusion of both.
    """
    prompt_source = getattr(cfg, "prompt_source", "manual_full")

    if prompt_source == "manual_full":
        return text_prompts

    if prompt_source == "manual_3d":
        return manual_3d_prompts

    if prompt_source == "llm_static":
        raise NotImplementedError(
            "llm_static is planned but not implemented yet. "
            "Use llm_dynamic_init first."
        )

    if prompt_source in {
        "llm_dynamic_init",
        "manualfull_llm_dynamic_init",
    }:
        if classnames is None:
            raise ValueError(
                f"Prompt source '{prompt_source}' requires classnames, "
                "but classnames=None was received."
            )

        llm_prompts = generate_llm_prompts(
            classnames=classnames,
            args=cfg,
            dataset_name=dataset_name,
        )

        if prompt_source == "llm_dynamic_init":
            return llm_prompts

        if prompt_source == "manualfull_llm_dynamic_init":
            static_template = text_prompts
        else:
            raise ValueError(f"Unexpected mixed prompt source: {prompt_source}")

        return {
            "__eemf_prompt_type__": "weighted_fusion",
            "static_template": static_template,
            "dynamic_template": llm_prompts,
            "static_weight": getattr(cfg, "prompt_static_weight", 0.75),
            "dynamic_weight": getattr(cfg, "prompt_dynamic_weight", 0.25),
        }

    raise ValueError(f"Unknown prompt source: {prompt_source}")
