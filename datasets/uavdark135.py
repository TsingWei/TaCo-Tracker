import os

import numpy as np

from datasets._core import BaseDataset, Sequence, SequenceList, load_text


def _uavdark_frame_key(filename):
    stem = os.path.splitext(filename)[0]
    if stem.isdigit():
        return (0, int(stem))
    return (1, filename)


class UAVDark135Dataset(BaseDataset):
    """UAVDark135 dataset.

    Expected layout::

        <uavdark135_path>/
        ├── anno_revise/           # or anno/
        │   ├── <sequence>.txt
        │   └── ...
        └── Sequences/             # or data_seq/
            ├── <sequence>/
            │   ├── 000001.jpg
            │   └── ...
            └── ...

    Also supports an extra ``UAVDark135_TSP_out/`` subdirectory.
    """

    def __init__(self, dataset_paths):
        super().__init__(dataset_paths)
        base = dataset_paths["uavdark135_path"]
        nested = os.path.join(base, "UAVDark135_TSP_out")
        if os.path.isdir(nested) and not os.path.isdir(os.path.join(base, "Sequences")):
            base = nested
        self.base_path = base
        self.seq_dir = self._pick_subdir("Sequences", "data_seq")
        self.anno_dir = self._pick_subdir("anno_revise", "anno")
        self.sequence_list = self._build_sequence_list()

    def _pick_subdir(self, *candidates):
        for name in candidates:
            path = os.path.join(self.base_path, name)
            if os.path.isdir(path):
                return path
        return os.path.join(self.base_path, candidates[0])

    def _build_sequence_list(self):
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(self.anno_dir)
            if name.endswith(".txt")
            and os.path.isdir(os.path.join(self.seq_dir, os.path.splitext(name)[0]))
        )

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.seq_dir, sequence_name)
        anno_path = os.path.join(self.anno_dir, f"{sequence_name}.txt")
        ground_truth_rect = load_text(anno_path, delimiter=(",", "\t", " "), dtype=np.float64).reshape(-1, 4)

        frame_names = sorted(
            [
                f
                for f in os.listdir(seq_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ],
            key=_uavdark_frame_key,
        )
        frames = [os.path.join(seq_path, frame_name) for frame_name in frame_names]
        # Use dataset='uav' so metrics.py applies the UAV-family convention for NaN/invalid GT frames.
        return Sequence(sequence_name, frames, "uav", ground_truth_rect, object_class=None)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

