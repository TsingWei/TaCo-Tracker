"""DataLoader construction shared by CPU preprocessing and DALI."""

import cv2
import torch
from torch.utils.data import DataLoader, DistributedSampler


def init_worker(worker_id):
    cv2.setNumThreads(0)


def build_torch_loader(dataset, batch_size, workers, device, training, rank, world_size,
                       seed, collate_fn=None, spawn=False):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=training, seed=seed, drop_last=training)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, sampler=sampler,
        drop_last=training, worker_init_fn=init_worker, collate_fn=collate_fn,
        pin_memory=device.type == "cuda" and collate_fn is None,
        generator=torch.Generator().manual_seed(seed + rank),
        multiprocessing_context="spawn" if workers and spawn else None,
    )
    if len(loader) == 0:
        raise ValueError("Use at least batch_size × world_size samples per training epoch")
    return loader
