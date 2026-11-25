#!/usr/bin/env python3
"""Train an official TaCo variant using a YAML recipe."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from config import default_data_root, resolve_data_roots, resolve_folder
from models import TACO_CONFIGS, build_taco
from train_lib.data.dali_pipeline import dali_available
from train_lib.data.datasets.coco_seq import MSCOCOSeq
from train_lib.data.datasets.got10k import Got10k
from train_lib.data.datasets.lasot import Lasot
from train_lib.data.datasets.tracking_net import TrackingNet
from train_lib.data.datasets.vasttrack import VastTrack
from train_lib.data.image_loader import jpeg4py_loader
from train_lib.data.loader import build_torch_loader
from train_lib.data.processing import PairProcessing
from train_lib.data.sampler import TrackingSampler
from train_lib.trainer import TaCoTrainer, seed_everything


REPO_ROOT = Path(__file__).resolve().parent
DATASETS = {
    "LASOT": ("lasot", Lasot, {"split": "train"}),
    "GOT10K_VOTTRAIN": ("got10k", Got10k, {"split": "vottrain"}),
    "GOT10K_VOTVAL": ("got10k", Got10k, {"split": "votval"}),
    "GOT10K_TRAIN": ("got10k", Got10k, {"split": "train"}),
    "GOT10K_VAL": ("got10k", Got10k, {"split": "val"}),
    "GOT10K_TRAIN_FULL": ("got10k", Got10k, {"split": "train_full"}),
    "COCO17": ("coco", MSCOCOSeq, {"version": "2017"}),
    "VASTTRACK": ("vasttrack", VastTrack, {"split": "train"}),
    "TRACKINGNET": ("trackingnet", TrackingNet, {}),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML recipe")
    parser.add_argument("--data-root", type=Path, default=default_data_root(),
                        help="Dataset parent directory ($TACO_DATA_ROOT or ./data)")
    parser.add_argument("--pretrained", help="Override backbone weights, or pass none for random initialization")
    parser.add_argument("--save-dir", type=Path, help="Output directory (outputs/<config_stem>)")
    parser.add_argument("--pipeline", choices=("auto", "torch", "dali"))
    parser.add_argument("--resume", help="Path to a full checkpoint, or auto for the latest checkpoint if present")
    for flag in ("epochs", "lr-drop-epoch", "batch-size", "samples-per-epoch", "num-workers", "seed", "print-interval", "save-interval"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA float16 autocast and scaling")
    parser.add_argument("--datasets", help="Comma-separated training datasets, with equal sampling ratios")
    args = parser.parse_args()
    if args.datasets is not None:
        args.datasets = [name.strip() for name in args.datasets.split(",")]
        unknown = [name for name in args.datasets if name.upper() not in DATASETS]
        if unknown:
            parser.error(f"Unknown training datasets {unknown}. Choose from {', '.join(DATASETS)}")
    return args


def load_config(args):
    with args.config.expanduser().open() as handle:
        config = yaml.safe_load(handle)
    train, data = config["train"], config["data"]
    for key in ("epochs", "lr_drop_epoch", "batch_size", "num_workers", "seed", "print_interval", "save_interval"):
        value = getattr(args, key)
        if value is not None:
            train[key] = value
    for key in ("epochs", "batch_size", "print_interval", "save_interval", "val_epoch_interval", "lr_drop_epoch"):
        if train[key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if train["num_workers"] < 0 or not 0 <= train["seed"] < 2**32:
        raise ValueError("num_workers must be nonnegative and seed must be an unsigned 32-bit integer")
    if args.num_workers is not None:
        config["dali"]["workers"] = args.num_workers
    if args.no_amp:
        train["amp"] = False
    if args.samples_per_epoch is not None:
        data["train"]["samples_per_epoch"] = args.samples_per_epoch
    if args.datasets is not None:
        data["train"].update(datasets=args.datasets, ratios=[1] * len(args.datasets))
    if args.pipeline is not None:
        data["pipeline"] = args.pipeline
    for name in ("template", "search"):
        variant = config["model"]["variant"]
        if data[name]["size"] != TACO_CONFIGS[variant][f"{name}_size"]:
            raise ValueError(f"{variant} requires the configured {name} size to match its architecture")
    return config


def training_device():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size


def choose_pipeline(requested, device):
    if requested == "torch":
        return requested
    if requested not in ("dali", "auto"):
        raise ValueError(f"Unknown data pipeline {requested!r}")
    available = device.type == "cuda" and dali_available()
    if requested == "dali" and not available:
        raise RuntimeError("--pipeline dali requires CUDA and a working NVIDIA DALI installation")
    return "dali" if available else "torch"


def build_datasets(names, roots, image_loader):
    datasets = []
    for name in names:
        key, constructor, options = DATASETS[name.upper()]
        root = roots[key]
        if key == "got10k" and not (root / "list.txt").is_file():
            root = resolve_folder(root, "train")
        if key in ("lasot", "vasttrack"):
            # Some downloads place class directories beneath an extra train/ split folder.
            # Guard against descending into the object class folder also named "train" / "Train".
            train_root = resolve_folder(root, "train")
            if train_root.is_dir() and not any(p.name.lower().startswith("train-") for p in train_root.iterdir()):
                if len(list(root.iterdir())) <= 5:
                    root = train_root
        if not root.is_dir():
            raise FileNotFoundError(f"{name} dataset directory missing: {root}. Set --data-root or TACO_DATA_ROOT.")
        datasets.append(constructor(root, image_loader=image_loader, **options))
    return datasets


def make_loader(config, roots, pipeline, device, rank, world_size, training):
    data, train = config["data"], config["train"]
    split = data["train" if training else "val"]
    use_dali = pipeline == "dali"
    if use_dali:
        from train_lib.data.dali_loader import DaliLoader, DaliProcessing, collate_dali, encoded_image

        image_loader = encoded_image
        processing = DaliProcessing(data, training)
        collate = collate_dali
        workers = config["dali"]["workers"]
    else:
        image_loader = jpeg4py_loader
        processing = PairProcessing(data, training)
        collate = None
        workers = train["num_workers"]

    datasets = build_datasets(split["datasets"], roots, image_loader)
    sampler = TrackingSampler(datasets, split["ratios"], split["samples_per_epoch"],
                              data["max_sample_interval"], processing)
    loader = build_torch_loader(sampler, train["batch_size"], workers, device, training,
                                rank, world_size, train["seed"], collate, spawn=use_dali)
    if use_dali:
        return DaliLoader(loader, train["batch_size"], device, data, train["seed"] + rank)
    return loader


def main():
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    config = load_config(args)
    model_cfg, train = config["model"], config["train"]
    save_dir = (args.save_dir or Path("outputs") / args.config.stem).expanduser()
    resume = Path(args.resume).expanduser() if args.resume not in (None, "auto") else None
    if args.resume == "auto" and (save_dir / "checkpoint_latest.pth").is_file():
        resume = save_dir / "checkpoint_latest.pth"

    device, rank, world_size = training_device()
    try:
        pipeline = choose_pipeline(config["data"]["pipeline"], device)
        seed_everything(train["seed"])
        model = build_taco(model_cfg["variant"], freeze_bn=model_cfg.get("freeze_bn", True))
        loaded = None
        pretrained = args.pretrained if args.pretrained is not None else model_cfg.get("pretrained_backbone")
        if resume is None and pretrained and pretrained.strip().lower() != "none":
            path = Path(pretrained).expanduser()
            if args.pretrained is None and not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"Backbone weights missing: {path}. Use --pretrained none for random initialization.")
            loaded = model.backbone.load_pretrained_backbone(path)
        model.to(device)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        trainer = TaCoTrainer(model, train, device, rank, world_size, save_dir)
        if resume is not None:
            trainer.resume(resume)
        elif loaded is not None:
            count, total = loaded
            trainer.log.info("Loaded %d/%d backbone tensors from %s (initialized %d BN counters)",
                             count, total, path, total - count)
        else:
            trainer.log.info("Backbone initialized from scratch")
        trainer.log.info("%s pipeline=%s device=%s processes=%d batch_per_process=%d amp=%s",
                         model_cfg["variant"], pipeline, device, world_size, train["batch_size"], trainer.amp)

        roots = resolve_data_roots(args.data_root)
        train_loader = make_loader(config, roots, pipeline, device, rank, world_size, training=True)
        trainer.fit(train_loader, lambda: make_loader(config, roots, pipeline, device, rank, world_size,
                                                       training=False))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
