import os
import numpy as np
from datasets._core import BaseDataset, Sequence, SequenceList, load_text


class GOT10KDataset(BaseDataset):
    """GOT-10k dataset.

    Download: http://got-10k.aitestunion.com/downloads

    Expected layout (example for val split)::

        <got10k_path>/
        └── val/
            ├── list.txt                     # sequence names, one per line
            ├── GOT-10k_Val_000001/
            │   ├── 00000001.jpg ...
            │   └── groundtruth.txt          # x,y,w,h per line, comma-separated
            └── ...
    """

    def __init__(self, dataset_paths, split='test'):
        super().__init__(dataset_paths)
        if split in ('test', 'val', 'hard'):
            self.base_path = os.path.join(dataset_paths["got10k_path"], split)
        else:
            self.base_path = os.path.join(dataset_paths["got10k_path"], 'train')
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = sorted(
            [f for f in os.listdir(frames_path) if f.endswith('.jpg')],
            key=lambda f: int(f[:-4]),
        )
        frames_list = [os.path.join(frames_path, f) for f in frame_list]

        return Sequence(sequence_name, frames_list, 'got10k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        if split == 'ltrval':
            raise NotImplementedError(
                'ltrval split requires a separate split file and is not supported in this release.'
            )
        with open('{}/list.txt'.format(self.base_path)) as f:
            return f.read().splitlines()
