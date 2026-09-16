#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BACKBONES=("ulip" "ulip2" "openshape" "uni3d")
DEFAULT_DATASETS=("modelnet" "scanobjnn" "modelnet_c" "scanobjnn_c")

usage() {
  echo "Usage: bash scripts/run_eemf_all.sh [GPU] [BACKBONES] [DATASETS]"
  echo "BACKBONES: all or comma list from: ${DEFAULT_BACKBONES[*]}"
  echo "DATASETS: all or comma list from: ${DEFAULT_DATASETS[*]}"
  echo "Example: bash scripts/run_eemf_all.sh 0"
  echo "Example: bash scripts/run_eemf_all.sh 1 ulip,ulip2 modelnet_c,scanobjnn_c"
}

contains_value() {
  local wanted="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

parse_selection() {
  local raw="$1"
  local output_name="$2"
  shift 2
  local defaults=("$@")
  local -n output_ref="${output_name}"

  output_ref=()
  if [[ -z "${raw}" || "${raw}" == "all" ]]; then
    output_ref=("${defaults[@]}")
    return 0
  fi

  local parts=()
  local part
  IFS=',' read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -z "${part}" ]]; then
      continue
    fi
    if ! contains_value "${part}" "${defaults[@]}"; then
      echo "ERROR: unsupported value '${part}'" >&2
      usage >&2
      exit 1
    fi
    output_ref+=("${part}")
  done

  if [[ "${#output_ref[@]}" -eq 0 ]]; then
    echo "ERROR: empty selection" >&2
    usage >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 3 ]]; then
  usage >&2
  exit 1
fi

gpu="${1:-${EEMF_GPU:-0}}"
backbone_arg="${2:-all}"
dataset_arg="${3:-all}"

selected_backbones=()
selected_datasets=()
parse_selection "${backbone_arg}" selected_backbones "${DEFAULT_BACKBONES[@]}"
parse_selection "${dataset_arg}" selected_datasets "${DEFAULT_DATASETS[@]}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
total=$(( ${#selected_backbones[@]} * ${#selected_datasets[@]} ))
index=0

echo "EEMF batch started: ${total} runs"

for backbone in "${selected_backbones[@]}"; do
  for dataset in "${selected_datasets[@]}"; do
    index=$((index + 1))
    echo "---- [${index}/${total}] ${backbone} ${dataset} ----"
    bash "${script_dir}/run_eemf.sh" "${backbone}" "${dataset}" "${gpu}"
  done
done

echo "EEMF batch finished: ${total} runs"
