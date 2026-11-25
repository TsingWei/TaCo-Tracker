import os
import numpy as np
from datasets._core import BaseDataset, Sequence, SequenceList, load_text


class VastTrackDataset(BaseDataset):
    """VastTrack test dataset.

    Expected layout::

        <vasttrack_test_path>/
        ├── <class_name>/
        │   ├── <class_name>-1/
        │   │   ├── imgs/
        │   │   │   ├── 00001.jpg ...
        │   │   └── Groundtruth.txt   # x,y,w,h, comma-separated
        │   └── ...
        └── ...
    """

    def __init__(self, dataset_paths, split=None):
        super().__init__(dataset_paths)
        if split != 'test':
            raise ValueError('Only the VastTrack test split is supported in this release.')
        self.base_path = dataset_paths["vasttrack_test_path"]
        self.sequence_list = self._build_sequence_list()

    def _build_sequence_list(self):
        sequence_list = []
        for class_name in sorted(os.listdir(self.base_path)):
            class_dir = os.path.join(self.base_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for seq_folder in sorted(os.listdir(class_dir)):
                seq_dir = os.path.join(class_dir, seq_folder)
                if os.path.isdir(seq_dir):
                    sequence_list.append(seq_folder)
        return sequence_list

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name[:sequence_name.rfind('-')]
        seq_path = os.path.join(self.base_path, class_name, sequence_name)

        anno_path = os.path.join(seq_path, 'Groundtruth.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        num_frames = ground_truth_rect.shape[0]
        frames_list = [
            os.path.join(seq_path, 'imgs', '{:05d}.jpg'.format(i))
            for i in range(1, num_frames + 1)
        ]

        return Sequence(sequence_name, frames_list, 'vasttrack',
                        ground_truth_rect.reshape(-1, 4), object_class=class_name)

    def __len__(self):
        return len(self.sequence_list)
