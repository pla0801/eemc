from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


RESULT_COLUMNS = [
    "dataset",
    "corruption",
    "cor_type",
    "backbone",
    "acc",
    "status",
]


def format_acc(value) -> str:
    if value is None or value == "":
        return ""
    return "{:.2f}".format(float(value))


def read_result_rows(path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _normalize_row(row: Dict[str, object]) -> Dict[str, str]:
    normalized = {column: str(row.get(column, "")) for column in RESULT_COLUMNS}
    normalized["acc"] = format_acc(normalized["acc"])
    return normalized


def _row_key(row: Dict[str, object]):
    return (
        str(row.get("dataset", "")),
        str(row.get("backbone", "")),
        str(row.get("cor_type", "")),
    )


def write_result_rows(path, rows: Iterable[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))
    tmp.replace(path)


def upsert_result_rows(path, rows: Iterable[Dict[str, object]]) -> None:
    existing = read_result_rows(path)
    positions = {_row_key(row): idx for idx, row in enumerate(existing)}
    merged = list(existing)
    for row in rows:
        normalized = _normalize_row(row)
        key = _row_key(normalized)
        if key in positions:
            merged[positions[key]] = normalized
        else:
            positions[key] = len(merged)
            merged.append(normalized)
    write_result_rows(path, merged)


def make_result_row(dataset_spec, backbone_spec, corruption_spec, acc, status: str = "done") -> Dict[str, object]:
    return {
        "dataset": dataset_spec.dataset_name,
        "corruption": corruption_spec.corruption,
        "cor_type": corruption_spec.cor_type,
        "backbone": backbone_spec.display_name,
        "acc": "" if acc is None else float(acc),
        "status": status,
    }
