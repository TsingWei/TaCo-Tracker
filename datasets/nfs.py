import os
import numpy as np
from datasets._core import BaseDataset, Sequence, SequenceList


class NFSDataset(BaseDataset):
    """Need For Speed (NFS) dataset.

    Download: http://ci2cv.net/nfs/index.html

    Expected layout::

        <nfs_path>/
        ├── <seq_name>/
        │   └── 240/
        │       ├── <seq_name>/       # frames at 240 fps
        │       │   ├── 00001.jpg ...
        │       └── <seq_name>.txt    # annotations at 240 fps
        └── ...

    Annotations have 9 columns; columns 1-4 are x1, y1, x2, y2.
    """

    VALID_FPS = (30, 60, 120, 240)

    def __init__(self, dataset_paths, fps=30):
        super().__init__(dataset_paths)
        if fps not in self.VALID_FPS:
            raise ValueError('fps must be one of {}; got {}'.format(self.VALID_FPS, fps))
        self.fps = fps
        self.step = 240 // fps
        self.base_path = dataset_paths["nfs_path"]
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        return sorted(
            d for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        )

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, seq_name):
        step = self.step
        anno_path = os.path.join(self.base_path, seq_name, '240', seq_name + '.txt')
        gt_240 = np.loadtxt(anno_path, usecols=range(9))
        n_gt = len(gt_240)
        gt_sub = gt_240[::step, 1:5].copy()
        gt_sub[:, 2] = gt_sub[:, 2] - gt_sub[:, 0]   # w = x2 - x1
        gt_sub[:, 3] = gt_sub[:, 3] - gt_sub[:, 1]   # h = y2 - y1

        frames_dir = os.path.join(self.base_path, seq_name, '240', seq_name)
        all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
        frames = [os.path.join(frames_dir, all_frames[i]) for i in range(0, n_gt, step)]

        return Sequence(seq_name, frames, 'nfs', gt_sub)

    def __len__(self):
        return len(self.sequence_list)
