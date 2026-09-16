#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/run_eemf.sh BACKBONE DATASET [GPU] [SEVERITY]"
  echo "BACKBONE: ulip, ulip2, openshape, uni3d"
  echo "DATASET: modelnet, scanobjnn, modelnet_c, scanobjnn_c"
  echo "SEVERITY: 0, 1, 2, 3, 4, or all; only valid for corrupted datasets"
  exit 0
fi

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "Usage: bash scripts/run_eemf.sh BACKBONE DATASET [GPU] [SEVERITY]"
  echo "BACKBONE: ulip, ulip2, openshape, uni3d"
  echo "DATASET: modelnet, scanobjnn, modelnet_c, scanobjnn_c"
  echo "SEVERITY: 0, 1, 2, 3, 4, or all; only valid for corrupted datasets"
  exit 1
fi

BACKBONE_KEY="$1"
DATASET_KEY="$2"
PHYSICAL_GPU="${3:-${EEMF_GPU:-0}}"
SEVERITY_VALUE="${4:-}"

case "${BACKBONE_KEY}" in
  ulip|ulip2|openshape|uni3d)
    ;;
  *)
    echo "ERROR: unsupported BACKBONE=${BACKBONE_KEY}"
    exit 1
    ;;
esac

case "${DATASET_KEY}" in
  modelnet|scanobjnn)
    if [[ -n "${SEVERITY_VALUE}" ]]; then
      echo "ERROR: SEVERITY is only valid for modelnet_c and scanobjnn_c"
      exit 1
    fi
    ;;
  modelnet_c|scanobjnn_c)
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET_KEY}"
    exit 1
    ;;
esac

SEVERITY_ARGS=()
if [[ -n "${SEVERITY_VALUE}" ]]; then
  case "${SEVERITY_VALUE}" in
    all)
      ;;
    0|1|2|3|4)
      SEVERITY_ARGS=(--severity "${SEVERITY_VALUE}")
      ;;
    *)
      echo "ERROR: SEVERITY must be 0, 1, 2, 3, 4, or all"
      exit 1
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

python -m eemf.run \
  --backbone "${BACKBONE_KEY}" \
  --dataset "${DATASET_KEY}" \
  "${SEVERITY_ARGS[@]}" \
  --result-root results \
  --config-dir configs \
  --prompt-cache-dir llm \
  --device 0 \
  --print-freq 500
