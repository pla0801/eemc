from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data_utils.modelnet_c import ModelNet_C
from data_utils.sonn_c import SONN_C


def build_test_data_loader(args):
    if args.dataset == "modelnet_c":
        dataset = ModelNet_C(args)
    elif args.dataset == "sonn_c":
        dataset = SONN_C(args)
    else:
        raise ValueError(f"Unsupported dataset loader: {args.dataset}")

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=max(0, int(getattr(args, "num_workers", 2))),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, dataset.classnames, dataset.template
