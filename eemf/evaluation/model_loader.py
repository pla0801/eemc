from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import open_clip
import torch
from omegaconf import OmegaConf

from models import openshape, ulip, uni3d


def _device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(getattr(args, "device", 0)))
    return torch.device("cpu")


def _require_file(path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return str(path)


def load_config(*yaml_files, cli_args=None, extra_args=None):
    cli_args = {} if cli_args is None else cli_args
    extra_args = [] if extra_args is None else extra_args
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    yaml_confs += [OmegaConf.from_cli(extra_args)]
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf


def load_uni3d(args):
    device = _device(args)
    text_pretrained = _require_file(args.pretrained)
    point_ckpt = _require_file(args.ckpt_path)

    open_clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=text_pretrained,
        device="cpu",
    )
    if hasattr(open_clip_model, "text"):
        open_clip_model.text.half().to(device)
    else:
        open_clip_model.half().to(device)
    open_clip_model.eval()

    lm3d_model = uni3d.create_uni3d(args)
    checkpoint = torch.load(point_ckpt, map_location="cpu")
    state_dict = checkpoint["module"]
    if not args.distributed and next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    lm3d_model.load_state_dict(state_dict)
    lm3d_model.half().to(device)
    lm3d_model.eval()
    return open_clip_model, lm3d_model


def load_openshape(args):
    device = _device(args)
    if args.oshape_version == "vitg14":
        clip_name = "ViT-bigG-14"
        text_pretrained = _require_file("weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin")
    elif args.oshape_version == "vitl14":
        clip_name = "ViT-L-14"
        text_pretrained = _require_file("weights/openshape/open_clip_pytorch_model/vit-l-14/laion2b_s32b_b82k.bin")
    else:
        raise NotImplementedError(f"Unsupported OpenShape version: {args.oshape_version}")

    open_clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained=text_pretrained)
    open_clip_model.half().to(device)
    open_clip_model.eval()

    config = load_config("models/openshape/config.yaml", cli_args=vars(args))
    lm3d_model = openshape.create_openshape(config)
    lm3d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lm3d_model)

    checkpoint = torch.load(_require_file(args.ckpt_path), map_location="cpu")
    model_dict = OrderedDict()
    if args.oshape_version == "vitg14":
        pattern = re.compile("module.")
        for key, value in checkpoint["state_dict"].items():
            if re.search("module", key):
                model_dict[re.sub(pattern, "", key)] = value
        lm3d_model.load_state_dict(model_dict)
    elif args.oshape_version == "vitl14":
        pattern = re.compile("pc_encoder.")
        for key, value in checkpoint.items():
            if re.search("pc_encoder", key):
                model_dict[re.sub(pattern, "", key)] = value
        lm3d_model.load_state_dict(model_dict)

    lm3d_model.half().to(device)
    lm3d_model.eval()
    return open_clip_model, lm3d_model


def load_ulip(args):
    device = _device(args)
    clip_model = ulip.create_clip_text_encoder(args)

    pretrain_slip = torch.load(_require_file(args.slip_ckpt_path), map_location="cpu")
    pretrain_slip_sd = pretrain_slip["state_dict"]
    pretrain_slip_sd = {k.replace("module.", ""): v for k, v in pretrain_slip_sd.items()}
    pretrain_slip_sd = {
        k: v
        for k, v in pretrain_slip_sd.items()
        if k.startswith("positional_embedding")
        or k.startswith("text_projection")
        or k.startswith("logit_scale")
        or k.startswith("transformer")
        or k.startswith("token_embedding")
        or k.startswith("ln_final")
    }
    clip_model.load_state_dict(OrderedDict(pretrain_slip_sd))
    clip_model.half().to(device)
    clip_model.eval()

    lm3d_model = ulip.create_ulip(args)
    pretrain_point = torch.load(_require_file(args.ckpt_path), map_location="cpu")
    pretrain_point_sd = pretrain_point["state_dict"]
    pretrain_point_sd = {k.replace("module.", ""): v for k, v in pretrain_point_sd.items()}
    pretrain_point_sd = {
        k: v
        for k, v in pretrain_point_sd.items()
        if k.startswith("pc_projection") or k.startswith("point_encoder")
    }
    lm3d_model.load_state_dict(OrderedDict(pretrain_point_sd))
    lm3d_model.half().to(device)
    lm3d_model.eval()
    return clip_model, lm3d_model


def load_models(args):
    if args.lm3d == "uni3d":
        return load_uni3d(args)
    if args.lm3d == "openshape":
        return load_openshape(args)
    if args.lm3d == "ulip":
        return load_ulip(args)
    raise NotImplementedError(f"Unsupported 3D backbone: {args.lm3d}")
