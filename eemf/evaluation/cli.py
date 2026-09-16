from __future__ import annotations

import argparse
from pathlib import Path

from eemf.evaluation.protocols import BACKBONE_KEYS, DATASET_KEYS, parse_severity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run EEMF evaluation.")

    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_KEYS))
    parser.add_argument("--dataset", required=True, choices=list(DATASET_KEYS))
    parser.add_argument("--severity", default=None, type=parse_severity)
    parser.add_argument("--result-root", default="results")

    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--print-freq", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--prompt-cache-dir", default="llm")
    parser.add_argument("--prompt-cache-file", default="")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-pro")
    parser.add_argument("--llm-api-base-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument("--llm-prompt-mode", default="multiview_2d3d", choices=[
        "pointcloud_geometry",
        "multiview_2d3d",
        "image4_pointcloud4_bridge2",
        "image10_pointcloud5",
        "image5_pointcloud10",
        "image12_pointcloud3",
        "multiview_2d3d_2to1",
        "multiview_2d3d_1to2",
    ])
    parser.add_argument("--dynamic-prompt-count", type=int, default=10)
    parser.add_argument("--prompt-static-weight", type=float, default=0.75)
    parser.add_argument("--prompt-dynamic-weight", type=float, default=0.25)
    parser.add_argument("--force-regenerate-prompts", action="store_true", default=False)
    parser.add_argument("--llm-max-retries", type=int, default=3)

    parser.add_argument("--npoints", default=1024, type=int)

    parser.add_argument("--ckpt-path", dest="ckpt_path", default="")
    parser.add_argument("--slip-ckpt-path", default="")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--pretrained-pc", default="")
    parser.add_argument("--pc-model", default="eva_giant_patch14_560")
    parser.add_argument("--clip-model", default="EVA02-E-14-plus")
    parser.add_argument("--drop-path-rate", default=0.0, type=float)

    parser.add_argument("--oshape-version", choices=["vitg14", "vitl14"], default="vitg14")
    parser.add_argument("--pc-feat-dim", type=int, default=768)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--num-group", type=int, default=512)
    parser.add_argument("--pc-encoder-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--patch-dropout", type=float, default=0.0)

    parser.add_argument("--ulip-version", choices=["ulip1", "ulip2"], default="")
    parser.add_argument("--pc-depth", type=int, default=12)
    parser.add_argument("--num-head", type=int, default=6)
    parser.add_argument("--encoder-dim", type=int, default=256)

    args = parser.parse_args(argv)
    return args


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
