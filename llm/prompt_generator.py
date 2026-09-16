from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - runtime fallback for minimal environments
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _arg_or_env(args, attr: str, env_name: str, default):
    value = getattr(args, attr, None)
    if value not in (None, ""):
        return value
    env_value = _env(env_name)
    if env_value:
        return env_value
    return default


def read_llm_api_key() -> str:
    api_key = _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError("LLM API key not found. Set LLM_API_KEY in the environment or in .env.")


def safe_name(name) -> str:
    name = str(name).replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def strip_json_code_fence(text) -> str:
    text = str(text).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + fence + r"(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(fence + r"$", "", text).strip()
    return text


def clean_prompt_text(text) -> str:
    text = str(text).strip().strip(",")
    text = text.replace('\\"', '"').strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text.strip().strip(",")


def parse_prompt_list(content):
    content = strip_json_code_fence(content)
    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]
        if isinstance(data, dict):
            for key in ["prompts", "descriptions", "sentences", "items", "data"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [clean_prompt_text(item) for item in value if clean_prompt_text(item)]
    except json.JSONDecodeError:
        pass

    left = content.find("[")
    right = content.rfind("]")
    if left != -1 and right != -1 and right > left:
        try:
            data = json.loads(content[left : right + 1])
            if isinstance(data, list):
                return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]
        except json.JSONDecodeError:
            pass

    prompts = []
    for line in content.splitlines():
        line = line.strip()
        line = re.sub(r"^[0-9]+[.)]\s*", "", line)
        line = line.strip("-• \t")
        if line and line not in {"[", "]", "{", "}"}:
            cleaned = clean_prompt_text(line)
            if cleaned:
                prompts.append(cleaned)
    return prompts


def _ratio_counts(prompt_count: int, prompt_mode: str):
    prompt_count = int(prompt_count)
    if prompt_mode in {"multiview_2d3d_2to1", "image10_pointcloud5"}:
        count_2d = round(prompt_count * 2 / 3)
        return count_2d, prompt_count - count_2d
    if prompt_mode in {"multiview_2d3d_1to2", "image5_pointcloud10"}:
        count_2d = round(prompt_count / 3)
        return count_2d, prompt_count - count_2d
    if prompt_mode == "image12_pointcloud3":
        if prompt_count != 15:
            raise ValueError("image12_pointcloud3 requires prompt_count=15.")
        return 12, 3
    raise ValueError(f"Prompt mode has no fixed 2D/3D ratio: {prompt_mode}")


def build_llm_request(classname, prompt_count, model, temperature, prompt_mode="multiview_2d3d"):
    if prompt_mode == "pointcloud_geometry":
        system_prompt = (
            "You generate concise English descriptions for 3D point cloud object recognition. "
            "Focus on geometric structure, object parts, shape, symmetry, and spatial layout. "
            "Do not describe colors, textures, photos, paintings, or camera styles. "
            "Return only a non-empty JSON array of strings."
        )
        user_prompt = (
            f"Generate exactly {prompt_count} different point-cloud-aware descriptions for the class "
            f"'{classname}'. Each description should be one complete sentence and should help "
            "a vision-language model recognize this object from a 3D point cloud. "
            "Return only a non-empty JSON array of strings. Do not return an empty array."
        )
    elif prompt_mode in {"multiview_2d3d", "image4_pointcloud4_bridge2"}:
        if prompt_mode == "image4_pointcloud4_bridge2" and int(prompt_count) != 10:
            raise ValueError("image4_pointcloud4_bridge2 requires prompt_count=10.")
        system_prompt = (
            "You generate concise English class descriptions for vision-language recognition of 3D point clouds. "
            "The whole description set should contain both 2D visual semantics and 3D point-cloud geometry, "
            "but each individual sentence does not need to contain both. "
            "Return only a JSON array of strings. Do not include explanations, numbering, markdown, or extra text."
        )
        if int(prompt_count) == 10:
            user_prompt = (
                f"Generate exactly 10 descriptions for the class '{classname}'. "
                "The 10 descriptions should be organized conceptually as follows, but return only one flat JSON array of 10 strings: "
                "Descriptions 1-4: 2D visual semantic descriptions, focusing on common visual appearance, recognizable parts, "
                "object identity, and image-level cues useful for CLIP-like text encoders. "
                "Descriptions 5-8: 3D point-cloud geometric descriptions, focusing on shape, structure, parts, symmetry, "
                "spatial layout, and point distribution. "
                "Descriptions 9-10: bridge descriptions that connect visual appearance with 3D geometric structure. "
                "Each description must be a complete English sentence. "
                "Avoid very short fragments. Do not output vague fragments such as 'a hand-carved stone'. "
                "Return only a JSON array of 10 strings."
            )
        else:
            user_prompt = (
                f"Generate exactly {prompt_count} descriptions for the class '{classname}'. "
                "Some descriptions should focus on 2D visual semantics and common appearance. "
                "Some descriptions should focus on 3D point-cloud geometry, shape, parts, symmetry, and spatial layout. "
                "A small number may connect both views. "
                "Each description must be a complete English sentence. "
                "Avoid very short fragments. Return only a JSON array of strings."
            )
    elif prompt_mode in {
        "multiview_2d3d_2to1",
        "multiview_2d3d_1to2",
        "image10_pointcloud5",
        "image5_pointcloud10",
        "image12_pointcloud3",
    }:
        if prompt_mode in {"image10_pointcloud5", "image5_pointcloud10", "image12_pointcloud3"} and int(prompt_count) != 15:
            raise ValueError(f"{prompt_mode} requires prompt_count=15.")
        count_2d, count_3d = _ratio_counts(prompt_count, prompt_mode)
        system_prompt = (
            "You generate concise English class descriptions for vision-language recognition of 3D point clouds. "
            "Return only a JSON array of strings. Do not include explanations, numbering, markdown, or extra text. "
            "Each sentence should be complete and useful for a CLIP-like text encoder aligned with 3D point-cloud features."
        )
        user_prompt = (
            f"Generate exactly {prompt_count} descriptions for the class '{classname}' with "
            f"{count_2d} image-style descriptions and {count_3d} pointcloud-style descriptions. "
            f"The first {count_2d} descriptions must focus on 2D visual semantics: common visual appearance, "
            "recognizable parts, object identity, image-level cues, and visual context. "
            f"The remaining {count_3d} descriptions must focus on 3D point-cloud geometry: shape, structure, "
            "parts, symmetry, spatial layout, and point distribution. "
            "Do not add bridge descriptions beyond these two groups. "
            "Each description must be a complete English sentence and at least eight words long. "
            f"Return only one flat JSON array of exactly {prompt_count} strings."
        )
    else:
        raise ValueError(f"Unknown LLM prompt mode: {prompt_mode}")

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max(1200, int(prompt_count) * 110),
    }


def call_openai_compatible_api(api_key, api_base_url, payload):
    request_data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_base_url,
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP error {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc

    response_json = json.loads(response_text)
    return response_json["choices"][0]["message"]["content"]


def get_prompt_cache_path(args, dataset_name):
    cache_dir = Path(getattr(args, "prompt_cache_dir", "llm"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    explicit_file = str(getattr(args, "prompt_cache_file", "") or "").strip()
    if explicit_file:
        explicit_path = Path(explicit_file)
        if explicit_path.is_absolute():
            explicit_path.parent.mkdir(parents=True, exist_ok=True)
            return explicit_path
        return cache_dir / explicit_path

    provider = safe_name(_arg_or_env(args, "llm_provider", "LLM_PROVIDER", "deepseek"))
    model_name = safe_name(_arg_or_env(args, "llm_model", "LLM_MODEL", "deepseek-v4-pro"))
    prompt_mode = safe_name(getattr(args, "llm_prompt_mode", "multiview_2d3d"))
    prompt_count = int(getattr(args, "dynamic_prompt_count", 10))
    dataset_name = safe_name(dataset_name or getattr(args, "dataset", "unknown_dataset"))

    return cache_dir / f"{dataset_name}_{provider}_{model_name}_{prompt_mode}_{prompt_count}_prompts.json"


def save_prompt_cache(cache_path, args, dataset_name, classnames, prompts, failed_classes=None):
    saved = {
        "prompt_source": getattr(args, "prompt_source", "llm_dynamic_init"),
        "llm_provider": _arg_or_env(args, "llm_provider", "LLM_PROVIDER", "deepseek"),
        "llm_model": _arg_or_env(args, "llm_model", "LLM_MODEL", "deepseek-v4-pro"),
        "llm_api_base_url": _arg_or_env(
            args,
            "llm_api_base_url",
            "LLM_API_BASE_URL",
            "https://api.deepseek.com/chat/completions",
        ),
        "llm_prompt_mode": getattr(args, "llm_prompt_mode", "multiview_2d3d"),
        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name or getattr(args, "dataset", "unknown_dataset"),
        "dynamic_prompt_count": int(getattr(args, "dynamic_prompt_count", 10)),
        "temperature": float(getattr(args, "llm_temperature", 0.3)),
        "class_names": classnames,
        "completed_class_names": sorted(prompts.keys()),
        "failed_classes": failed_classes or [],
        "prompts": prompts,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)


def load_existing_prompts(cache_path):
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    prompts = saved.get("prompts", saved)
    return prompts if isinstance(prompts, dict) else {}


def is_valid_prompt_text(text) -> bool:
    text = str(text).strip()
    words = text.split()
    if len(words) < 8:
        return False
    if not text.endswith((".", "!", "?")):
        return False
    return text.lower().strip(" .,!?:;") not in {"a hand-carved stone", "hand-carved stone"}


def generate_one_class_prompts(classname, args, api_key, api_base_url, model, prompt_count, temperature, prompt_mode):
    max_retries = int(getattr(args, "llm_max_retries", 3))
    last_error = None

    for attempt in range(1, max_retries + 1):
        payload = build_llm_request(classname, prompt_count, model, temperature, prompt_mode)
        try:
            content = call_openai_compatible_api(api_key, api_base_url, payload)
        except Exception as exc:
            last_error = exc
            print(f"LLM request failed for {classname}. Retry {attempt}/{max_retries}. Error: {exc}", flush=True)
            time.sleep(min(10.0, 2.0 * attempt))
            continue

        prompts = [p for p in parse_prompt_list(content) if is_valid_prompt_text(p)]
        if len(prompts) >= prompt_count:
            return prompts[:prompt_count]

        print(
            f"LLM returned {len(prompts)}/{prompt_count} valid prompts for {classname}. "
            f"Retry {attempt}/{max_retries}.",
            flush=True,
        )
        time.sleep(1.0)

    if last_error is not None:
        raise RuntimeError(f"LLM request failed after {max_retries} retries for class: {classname}") from last_error
    raise RuntimeError(f"LLM returned no valid prompts for class: {classname}")


def generate_llm_prompts(classnames, args, dataset_name=None):
    classnames = [str(name).replace("_", " ") for name in classnames]
    cache_path = get_prompt_cache_path(args, dataset_name)
    force_regenerate = bool(getattr(args, "force_regenerate_prompts", False))
    prompt_count = int(getattr(args, "dynamic_prompt_count", 10))

    existing_prompts = {} if force_regenerate else load_existing_prompts(cache_path)
    missing_classes = [
        classname
        for classname in classnames
        if len(existing_prompts.get(classname, [])) < prompt_count
    ]

    if not missing_classes:
        return existing_prompts

    api_key = read_llm_api_key()
    api_base_url = _arg_or_env(
        args,
        "llm_api_base_url",
        "LLM_API_BASE_URL",
        "https://api.deepseek.com/chat/completions",
    )
    model = _arg_or_env(args, "llm_model", "LLM_MODEL", "deepseek-v4-pro")
    temperature = float(_arg_or_env(args, "llm_temperature", "LLM_TEMPERATURE", 0.3))
    prompt_mode = getattr(args, "llm_prompt_mode", "multiview_2d3d")

    failed_classes = []
    for idx, classname in enumerate(missing_classes, start=1):
        print(f"Generating LLM prompts [{idx}/{len(missing_classes)}]: {classname}", flush=True)
        try:
            class_prompts = generate_one_class_prompts(
                classname,
                args,
                api_key,
                api_base_url,
                model,
                prompt_count,
                temperature,
                prompt_mode,
            )
        except Exception:
            failed_classes.append(classname)
            save_prompt_cache(cache_path, args, dataset_name, classnames, existing_prompts, failed_classes)
            raise

        existing_prompts[classname] = class_prompts
        save_prompt_cache(cache_path, args, dataset_name, classnames, existing_prompts, failed_classes)

    return existing_prompts
