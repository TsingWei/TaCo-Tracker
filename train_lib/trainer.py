"""Supervised TaCo training with optional CUDA AMP and distributed execution."""

import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from .losses import TaCoLoss, box_xywh_to_xyxy


METRICS = ("Loss/total", "Loss/giou", "Loss/l1", "Loss/focal", "IoU")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _logger(save_dir, rank):
    logger = logging.getLogger(f"taco.train.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(save_dir / "train.log")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def _summary_writer(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            return None
    return SummaryWriter(str(path))


def _save_atomic(payload, destination):
    temp = destination.with_suffix(".tmp")
    try:
        torch.save(payload, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


class TaCoTrainer:
    def __init__(self, model, config, device, rank, world_size, save_dir):
        self.model = model
        self.net = model.module if isinstance(model, DistributedDataParallel) else model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.save_dir = Path(save_dir)
        if rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log = _logger(self.save_dir, rank)
        self.writer = _summary_writer(self.save_dir / "tensorboard") if rank == 0 else None

        self.amp = config["amp"] and device.type == "cuda"
        if config.get("amp_dtype", "float16") != "float16":
            raise ValueError("TaCo training supports float16 AMP")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.nan_skip = config["nan_skip"] and self.amp
        self.criterion = TaCoLoss(config["giou_weight"], config["l1_weight"], config["focal_weight"])

        body, fresh = [], []
        for name, parameter in self.net.named_parameters():
            if parameter.requires_grad:
                (fresh if self.net.fresh_param_predicate(name) else body).append(parameter)
        lr = config["lr"]
        self.optimizer = torch.optim.AdamW(
            [{"params": fresh}, {"params": body, "lr": lr * config["backbone_multiplier"]}],
            lr=lr, weight_decay=config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config["lr_drop_epoch"], gamma=config["lr_decay_rate"],
        )
        self.epoch = 0
        self.step = 0

    def resume(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["model"] != self.net.variant_name:
            raise ValueError(f"{path} was saved for {checkpoint['model']}, expected {self.net.variant_name}")
        self.net.load_state_dict(checkpoint["net"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.amp and checkpoint["scaler"]:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.log.info("Resumed %s at epoch %d after %d steps", path, self.epoch, self.step)

    def _save_checkpoint(self, final):
        weights = {"net": self.net.state_dict(), "model": self.net.variant_name, "epoch": self.epoch}
        checkpoint = dict(weights, optimizer=self.optimizer.state_dict(),
                          scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
                          step=self.step)
        if self.epoch % self.config["save_interval"] == 0:
            _save_atomic(checkpoint, self.save_dir / f"checkpoint_ep{self.epoch:04d}.pth")
        if final:
            path = self.save_dir / f"{self.net.variant_name}_ep{self.epoch:04d}.pth"
            _save_atomic(weights, path)
            self.log.info("Saved inference weights to %s", path)
        _save_atomic(checkpoint, self.save_dir / "checkpoint_latest.pth")

    def _bn_snapshot(self):
        return [
            (module, module.running_mean.clone(), module.running_var.clone(),
             module.num_batches_tracked.clone())
            for module in self.net.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and module.training and module.track_running_stats
        ]

    def _skip_nan_batch(self, snapshot, outputs):
        # Rebind buffers. Backward may still hold the previous tensors.
        with torch.no_grad():
            for module, mean, var, count in snapshot:
                module.running_mean = mean
                module.running_var = var
                module.num_batches_tracked = count
        if self.world_size > 1:
            # Complete the DDP backward on every rank, then discard these gradients.
            zero_loss = sum((torch.nan_to_num(output.float()) * 0).sum()
                            for output in outputs.values())
            zero_loss.backward()
        self.optimizer.zero_grad(set_to_none=True)

    def _batch(self, batch, training):
        # Keep the labels on the host for the reference Gaussian rasterizer.
        search_box = batch["search_anno"]
        template = batch["template_images"].to(self.device, non_blocking=True)
        search = batch["search_images"].to(self.device, non_blocking=True)
        template_box = box_xywh_to_xyxy(batch["template_anno"]).to(self.device, non_blocking=True)
        snapshot = self._bn_snapshot() if training and self.nan_skip else []
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            outputs = self.model(template, search, template_box)

        finite = torch.stack([tensor.detach().isfinite().all() for tensor in outputs.values()]).all()
        if finite.item():
            loss, metrics = self.criterion(outputs, search_box)
            finite = loss.detach().isfinite()
        if self.world_size > 1:
            finite = finite.to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            if not (training and self.nan_skip):
                raise FloatingPointError("Non-finite TaCo output or loss")
            self._skip_nan_batch(snapshot, outputs)
            return None

        if training:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.net.parameters(), self.config["grad_clip_norm"],
                            error_if_nonfinite=not self.amp)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return metrics

    def _log_progress(self, totals, phase, epoch, iteration, batches, started):
        if self.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if self.rank != 0:
            return
        values = totals.tolist()
        valid_samples, seen, skipped = values[-3:]
        elapsed = max(time.monotonic() - started, 1e-9)
        stats = {key: value / valid_samples for key, value in zip(METRICS, values)} if valid_samples else {}
        stats["LR"] = self.optimizer.param_groups[0]["lr"]
        stats["FPS"] = seen / elapsed
        remaining = elapsed / iteration * (batches - iteration)
        summary = " ".join(f"{key}={value:.5g}" for key, value in stats.items())
        self.log.info("%s epoch %d [%d/%d] %s ETA=%s skipped=%d", phase, epoch, iteration, batches,
                      summary, timedelta(seconds=int(remaining)), skipped // self.world_size)
        if self.writer is not None:
            step = self.step if phase == "train" else epoch
            for key, value in stats.items():
                self.writer.add_scalar(f"{phase}/{key}", value, step)

    def _run_epoch(self, loader, epoch, training):
        self.model.train(training)
        if loader.sampler is not None and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        loader.generator.manual_seed(self.config["seed"] + epoch * self.world_size + self.rank)
        totals = torch.zeros(len(METRICS) + 3, device=self.device, dtype=torch.float64)
        started = time.monotonic()
        batches = len(loader)
        local_valid_samples = 0
        with nullcontext() if training else torch.no_grad():
            for iteration, batch in enumerate(loader, 1):
                metrics = self._batch(batch, training)
                count = len(batch["search_anno"])
                totals[-2] += count
                if metrics is None:
                    totals[-1] += 1
                else:
                    totals[:-3] += torch.stack([metrics[key] for key in METRICS]) * count
                    totals[-3] += count
                    local_valid_samples += count
                if training:
                    self.step += 1
                if iteration % self.config["print_interval"] == 0 or iteration == batches:
                    self._log_progress(totals.clone(), "train" if training else "val",
                                       epoch, iteration, batches, started)
        if not local_valid_samples:
            raise FloatingPointError(f"Epoch {epoch} produced no finite batches")

    def fit(self, train_loader, val_loader_factory):
        val_loader = None
        try:
            for epoch in range(self.epoch + 1, self.config["epochs"] + 1):
                # Derive worker and process seeds from epoch so an epoch-boundary
                # restart samples the same next epoch as an uninterrupted run.
                seed = self.config["seed"] + epoch * self.world_size
                seed_everything(seed + self.rank)
                self._run_epoch(train_loader, epoch, training=True)
                if epoch % self.config["val_epoch_interval"] == 0:
                    if val_loader is None:
                        val_loader = val_loader_factory()
                    self._run_epoch(val_loader, epoch, training=False)
                self.scheduler.step()
                self.epoch = epoch
                if self.rank == 0:
                    self._save_checkpoint(final=epoch == self.config["epochs"])
        finally:
            if self.writer is not None:
                self.writer.close()
            for handler in self.log.handlers[:]:
                self.log.removeHandler(handler)
                handler.close()
