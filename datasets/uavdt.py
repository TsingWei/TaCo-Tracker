import os

import numpy as np

from datasets._core import BaseDataset, Sequence, SequenceList, load_text


class UAVDTDataset(BaseDataset):
    """UAVDT-SOT dataset.

    Expected layout::

        <uavdt_path>/
        ├── anno/
        │   ├── S0101_gt.txt
        │   └── ...
        └── sequences/
            ├── S0101/
            │   ├── img000001.jpg
            │   └── ...
            └── ...
    """

    def __init__(self, dataset_paths):
        super().__init__(dataset_paths)
        self.base_path = dataset_paths["uavdt_path"]
        self.sequence_list = self._build_sequence_list()

    def _build_sequence_list(self):
        seq_root = os.path.join(self.base_path, 'sequences')
        return sorted([name for name in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, name))])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, 'sequences', sequence_name)
        anno_path = os.path.join(self.base_path, 'anno', f'{sequence_name}_gt.txt')
        ground_truth_rect = load_text(anno_path, delimiter=(",", "\t", " "), dtype=np.float64).reshape(-1, 4)

        frames = [
            os.path.join(seq_path, frame_name)
            for frame_name in sorted(os.listdir(seq_path))
            if frame_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        return Sequence(sequence_name, frames, 'uavdt', ground_truth_rect, object_class=None)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)