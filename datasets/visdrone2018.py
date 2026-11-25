import os

import numpy as np

from datasets._core import BaseDataset, Sequence, SequenceList, load_text


def _visdrone_frame_key(filename):
    stem = os.path.splitext(filename)[0]
    if stem.startswith("img") and stem[3:].isdigit():
        return (0, int(stem[3:]))
    if stem.isdigit():
        return (0, int(stem))
    return (1, filename)


class VisDrone2018Dataset(BaseDataset):
    """VisDrone2018-SOT dataset.

    Expected layout::

        <visdrone2018_path>/
        ├── annotations/
        │   ├── <sequence>.txt
        │   └── ...
        └── sequences/
            ├── <sequence>/
            │   ├── img0000001.jpg
            │   └── ...
            └── ...
    """

    def __init__(self, dataset_paths):
        super().__init__(dataset_paths)
        self.base_path = dataset_paths["visdrone2018_path"]
        self.sequence_list = self._build_sequence_list()

    def _build_sequence_list(self):
        seq_root = os.path.join(self.base_path, "sequences")
        anno_root = os.path.join(self.base_path, "annotations")
        return sorted(
            name
            for name in os.listdir(seq_root)
            if os.path.isdir(os.path.join(seq_root, name))
            and os.path.isfile(os.path.join(anno_root, f"{name}.txt"))
        )

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, "sequences", sequence_name)
        anno_path = os.path.join(self.base_path, "annotations", f"{sequence_name}.txt")
        ground_truth_rect = load_text(anno_path, delimiter=(",", "\t", " "), dtype=np.float64).reshape(-1, 4)

        frame_names = sorted(
            [
                f
                for f in os.listdir(seq_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ],
            key=_visdrone_frame_key,
        )
        frames = [os.path.join(seq_path, frame_name) for frame_name in frame_names]
        return Sequence(sequence_name, frames, "visdrone2018", ground_truth_rect, object_class=None)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

