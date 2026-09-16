from __future__ import annotations

import gc
import os
import traceback
from pathlib import Path

from eemf.evaluation.cli import parse_args, project_root
from eemf.evaluation.logging_utils import (
    append_log_footer,
    append_log_header,
    append_log_line,
    redirect_to_log,
)
from eemf.evaluation.protocols import build_dataset_spec, get_backbone_spec, run_dir_name
from eemf.evaluation.result_writer import make_result_row, upsert_result_rows


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _precheck_data_files(root: Path, dataset_spec, backbone_spec, result_file: Path):
    missing = []
    for corruption_spec in dataset_spec.corruption_specs:
        data_file = _resolve_under_root(root, corruption_spec.file_path)
        if not data_file.exists():
            missing.append((corruption_spec, data_file))

    if not missing:
        return

    upsert_result_rows(
        result_file,
        [
            make_result_row(dataset_spec, backbone_spec, corruption_spec, None, "missing_file")
            for corruption_spec, _data_file in missing
        ],
    )
    missing_text = "\n".join(str(path.relative_to(root)) for _spec, path in missing)
    raise FileNotFoundError(f"Missing data files:\n{missing_text}")


def _free_stream_objects(test_loader) -> None:
    import torch

    try:
        if test_loader is not None:
            del test_loader
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_one_stream(
    args,
    cfg,
    clip_model,
    lm3d_model,
    clip_weights_state,
    dataset_spec,
    backbone_spec,
    corruption_spec,
    result_file: Path,
    log_file: Path,
    stream_index: int,
    stream_count: int,
):
    from eemf.evaluation.model_setup import (
        build_clip_weights_once,
        build_release_loader,
        build_text_distribution_once,
        build_text_distribution_template,
        set_release_seed,
    )
    from eemf.method.engine import run_stream

    args.cor_type = corruption_spec.cor_type
    test_loader = None
    stream_label = f"{dataset_spec.dataset_name}/{corruption_spec.cor_type}"

    try:
        print(f"---- stream {stream_index}/{stream_count}: {stream_label} ----", flush=True)
        with redirect_to_log(log_file):
            append_log_line(log_file, f"stream started: {stream_label}")
            set_release_seed(args.seed)
            test_loader, classnames, template = build_release_loader(args)
            clip_weights = build_clip_weights_once(args, clip_model, clip_weights_state, classnames, template)

            if clip_weights_state.get("text_dist") is None:
                text_template = build_text_distribution_template(args, classnames, template)
            else:
                text_template = template
            text_dist = build_text_distribution_once(args, clip_model, clip_weights_state, classnames, text_template)

        with redirect_to_log(log_file, tee_stdout=True):
            result = run_stream(
                args,
                cfg["positive"],
                cfg["negative"],
                test_loader,
                lm3d_model,
                clip_weights,
                text_dist=text_dist,
            )

        append_log_line(log_file, f"stream finished: {stream_label}, acc={result.acc:.2f}, total={result.total}")
        upsert_result_rows(
            result_file,
            [make_result_row(dataset_spec, backbone_spec, corruption_spec, result.acc, "done")],
        )
        return result

    except Exception:
        with redirect_to_log(log_file):
            append_log_line(log_file, f"stream failed: {stream_label}")
            traceback.print_exc()
        upsert_result_rows(
            result_file,
            [make_result_row(dataset_spec, backbone_spec, corruption_spec, None, "failed")],
        )
        raise

    finally:
        _free_stream_objects(test_loader)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from eemf.evaluation.model_setup import (
        apply_runtime_defaults,
        backbone_matches,
        get_backbone_name,
        load_release_config,
        load_release_models,
        set_release_seed,
        validate_text_prompt_cache,
    )

    root = project_root()
    backbone_spec = get_backbone_spec(args.backbone)
    dataset_spec = build_dataset_spec(args.dataset, severity=args.severity)
    apply_runtime_defaults(args, dataset_spec, backbone_spec)

    actual_backbone = get_backbone_name(args)
    if not backbone_matches(backbone_spec, actual_backbone):
        raise ValueError(f"Requested {backbone_spec.display_name}, but model arguments describe {actual_backbone}")

    result_root = _resolve_under_root(root, args.result_root)
    run_dir = result_root / run_dir_name(backbone_spec, dataset_spec)
    result_file = run_dir / "result.csv"
    log_file = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(root)
    append_log_header(log_file)

    try:
        _precheck_data_files(root, dataset_spec, backbone_spec, result_file)
        with redirect_to_log(log_file):
            prompt_cache = validate_text_prompt_cache(args, dataset_spec.prompt_dataset, root)
            if prompt_cache:
                append_log_line(log_file, f"text prompt cache: {prompt_cache.relative_to(root)}")

        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.device))

        print(f"---- loading {backbone_spec.display_name} models ----", flush=True)
        with redirect_to_log(log_file):
            append_log_line(log_file, f"loading models: {backbone_spec.display_name}")
            set_release_seed(args.seed)
            clip_model, lm3d_model = load_release_models(args)
            cfg = load_release_config(args)
            append_log_line(log_file, "models ready")
        print("---- models ready ----", flush=True)

        clip_weights_state = {
            "clip_weights": None,
            "classnames": None,
            "template": None,
            "text_dist": None,
        }
        stream_count = len(dataset_spec.corruption_specs)
        for stream_index, corruption_spec in enumerate(dataset_spec.corruption_specs, start=1):
            _evaluate_one_stream(
                args=args,
                cfg=cfg,
                clip_model=clip_model,
                lm3d_model=lm3d_model,
                clip_weights_state=clip_weights_state,
                dataset_spec=dataset_spec,
                backbone_spec=backbone_spec,
                corruption_spec=corruption_spec,
                result_file=result_file,
                log_file=log_file,
                stream_index=stream_index,
                stream_count=stream_count,
            )

        append_log_footer(log_file, "done")
        print(f"Results written to {result_file.relative_to(root)}", flush=True)
        return 0

    except Exception:
        append_log_footer(log_file, "failed")
        raise

    finally:
        os.chdir(old_cwd)
