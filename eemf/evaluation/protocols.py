from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


BACKBONE_KEYS: Tuple[str, ...] = ("ulip", "ulip2", "openshape", "uni3d")
DATASET_KEYS: Tuple[str, ...] = ("modelnet", "scanobjnn", "modelnet_c", "scanobjnn_c")
CORRUPTIONS: Tuple[str, ...] = (
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "rotate",
    "scale",
    "jitter",
)
SEVERITIES: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class CorruptionSpec:
    corruption: str
    cor_type: str
    file_path: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    dataset_name: str
    data_root_attr: str
    data_root: str
    dataset_token: str
    prompt_dataset: str
    corruption_specs: List[CorruptionSpec]
    setup_args: Optional[Callable[[object], None]] = None


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    token: str
    display_name: str


_BACKBONES = {
    "ulip": BackboneSpec("ulip", "ulip", "ULIP"),
    "ulip2": BackboneSpec("ulip2", "ulip2", "ULIP-2"),
    "openshape": BackboneSpec("openshape", "openshape", "OpenShape"),
    "uni3d": BackboneSpec("uni3d", "uni3d", "Uni3D"),
}


def normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def parse_severity(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "" or text == "all":
        return None
    try:
        severity = int(text)
    except ValueError as exc:
        raise ValueError("severity must be one of 0,1,2,3,4 or all") from exc
    if severity not in SEVERITIES:
        raise ValueError("severity must be one of 0,1,2,3,4 or all")
    return severity


def get_backbone_spec(backbone_key: str) -> BackboneSpec:
    key = normalize_key(backbone_key)
    if key not in _BACKBONES:
        raise ValueError("Unsupported backbone {!r}; expected {}".format(backbone_key, ", ".join(BACKBONE_KEYS)))
    return _BACKBONES[key]


def _setup_scanobjnn_args(args) -> None:
    args.sonn_variant = "hardest"


def _clean_stream(path: str) -> CorruptionSpec:
    return CorruptionSpec("clean", "clean", path)


def _corrupted_streams(data_root: str, severity: Optional[int], prefix: Optional[str] = None) -> List[CorruptionSpec]:
    severities: Sequence[int] = SEVERITIES if severity is None else (severity,)
    specs: List[CorruptionSpec] = []
    for corruption in CORRUPTIONS:
        for sev in severities:
            cor_type = f"{corruption}_{sev}"
            if prefix:
                path = Path(data_root) / prefix / f"{cor_type}.h5"
            else:
                path = Path(data_root) / f"{cor_type}.h5"
            specs.append(CorruptionSpec(corruption, cor_type, str(path)))
    return specs


def build_dataset_spec(dataset_key: str, severity=None) -> DatasetSpec:
    key = normalize_key(dataset_key)
    sev = parse_severity(severity)

    if key == "modelnet":
        if sev is not None:
            raise ValueError("severity is only valid for corrupted datasets")
        data_root = "data/modelnet_c"
        return DatasetSpec(
            dataset_key="modelnet",
            dataset_name="modelnet_c",
            data_root_attr="modelnet_c_root",
            data_root=data_root,
            dataset_token="modelnet_clean",
            prompt_dataset="modelnet_c",
            corruption_specs=[_clean_stream(str(Path(data_root) / "clean.h5"))],
        )

    if key == "scanobjnn":
        if sev is not None:
            raise ValueError("severity is only valid for corrupted datasets")
        data_root = "data/sonn_c"
        return DatasetSpec(
            dataset_key="scanobjnn",
            dataset_name="sonn_c",
            data_root_attr="sonn_c_root",
            data_root=data_root,
            dataset_token="scanobjnn_hardest_clean",
            prompt_dataset="sonn_c",
            corruption_specs=[_clean_stream(str(Path(data_root) / "hardest" / "clean.h5"))],
            setup_args=_setup_scanobjnn_args,
        )

    if key in {"modelnet_c", "modelnetc"}:
        data_root = "data/modelnet_c"
        return DatasetSpec(
            dataset_key="modelnet_c",
            dataset_name="modelnet_c",
            data_root_attr="modelnet_c_root",
            data_root=data_root,
            dataset_token="modelnetc",
            prompt_dataset="modelnet_c",
            corruption_specs=_corrupted_streams(data_root, sev),
        )

    if key in {"scanobjnn_c", "scanobjnnc"}:
        data_root = "data/sonn_c"
        return DatasetSpec(
            dataset_key="scanobjnn_c",
            dataset_name="sonn_c",
            data_root_attr="sonn_c_root",
            data_root=data_root,
            dataset_token="scanobjnnc_hardest",
            prompt_dataset="sonn_c",
            corruption_specs=_corrupted_streams(data_root, sev, prefix="hardest"),
            setup_args=_setup_scanobjnn_args,
        )

    raise ValueError("Unsupported dataset {!r}; expected {}".format(dataset_key, ", ".join(DATASET_KEYS)))


def apply_dataset_args(args, dataset_spec: DatasetSpec) -> None:
    args.dataset = dataset_spec.dataset_name
    setattr(args, dataset_spec.data_root_attr, dataset_spec.data_root)
    args.data_root = dataset_spec.data_root
    args.cor_type = dataset_spec.corruption_specs[0].cor_type
    if dataset_spec.setup_args is not None:
        dataset_spec.setup_args(args)


def run_dir_name(backbone_spec: BackboneSpec, dataset_spec: DatasetSpec) -> str:
    return f"{backbone_spec.token}_{dataset_spec.dataset_token}"
