import os

import numpy as np

from datasets._core import BaseDataset, Sequence, SequenceList, load_text


class DTB70Dataset(BaseDataset):
    """DTB70 dataset.

    Expected layout::

        <dtb70_path>/
        ├── Animal1/
        │   ├── groundtruth_rect.txt
        │   └── img/
        │       ├── 000001.jpg
        │       └── ...
        └── ...
    """

    def __init__(self, dataset_paths):
        super().__init__(dataset_paths)
        self.base_path = dataset_paths["dtb70_path"]
        self.sequence_list = self._build_sequence_list()

    def _build_sequence_list(self):
        return sorted([name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)
        anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        ground_truth_rect = load_text(anno_path, delimiter=(",", "\t", " "), dtype=np.float64).reshape(-1, 4)

        img_dir = os.path.join(seq_path, 'img')
        frames = [
            os.path.join(img_dir, frame_name)
            for frame_name in sorted(os.listdir(img_dir))
            if frame_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        return Sequence(sequence_name, frames, 'dtb70', ground_truth_rect, object_class=None)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)