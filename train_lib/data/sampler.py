"""Sample a causal visible template/search pair or two crops of a COCO image."""

import bisect
import random

import torch
from torch.utils.data import Dataset

from .datasets.anno_utils import SequenceInfoCache
from .tensor import TensorDict


class TrackingSampler(Dataset):
    def __init__(self, datasets, ratios, samples_per_epoch, max_gap, processing):
        if len(datasets) != len(ratios) or not datasets or any(r <= 0 for r in ratios):
            raise ValueError("Each dataset needs a positive sampling ratio")
        empty = [dataset.get_name() for dataset in datasets if not dataset.get_num_sequences()]
        if empty:
            raise ValueError(f"Training datasets contain no sequences: {empty}")
        if samples_per_epoch < 1 or max_gap < 1:
            raise ValueError("Sample count and maximum frame interval must be positive")
        self.datasets = datasets
        self.ratios = ratios
        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.processing = processing
        self.cache = SequenceInfoCache()

    def __len__(self):
        return self.samples_per_epoch

    def _pair(self, visible):
        frames = visible.nonzero(as_tuple=True)[0].tolist()
        if len(visible) < 20 or len(frames) <= 4:
            return None
        candidates = [
            i
            for i, frame in enumerate(frames[:-1])
            if frames[i + 1] - frame <= self.max_gap
        ]
        if not candidates:
            return None
        first = random.choice(candidates)
        stop = bisect.bisect_right(frames, frames[first] + self.max_gap, lo=first + 1)
        return frames[first], frames[random.randrange(first + 1, stop)]

    def __getitem__(self, index):
        last_error = None
        dataset_id = random.choices(range(len(self.datasets)), weights=self.ratios)[0]
        dataset = self.datasets[dataset_id]
        for _ in range(200):
            seq = random.randrange(dataset.get_num_sequences())
            try:
                key = (dataset_id, seq)
                info = self.cache.get(key)
                if info is None:
                    info = dataset.get_sequence_info(seq)
                    self.cache.put(key, info)
                boxes = info["bbox"]
                valid = torch.isfinite(boxes).all(dim=-1)
                valid &= (boxes[:, 2:] > 0).all(dim=-1)
                if dataset.is_video_sequence():
                    pair = self._pair(valid & info["visible"].bool())
                else:
                    pair = (0, 0) if valid[0] else None
                if pair is None:
                    continue
                frames, annotations, _ = dataset.get_frames(seq, pair, info)
                return self.processing(TensorDict(
                    template_images=frames[0], search_images=frames[1],
                    template_anno=annotations["bbox"][0], search_anno=annotations["bbox"][1],
                ))
            except (OSError, ValueError, IndexError) as exc:
                last_error = exc
        raise RuntimeError(f"{dataset.get_name()}: no valid tracking pair after 200 attempts") from last_error
