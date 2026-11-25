import os
import numpy as np
from datasets._core import BaseDataset, Sequence, SequenceList, load_text


class TrackingNetDataset(BaseDataset):
    """TrackingNet test set.

    Download: https://tracking-net.org/

    Expected layout::

        <trackingnet_path>/
        └── TEST/
            ├── anno/
            │   ├── <seq_name>.txt   # x,y,w,h, comma-separated
            │   └── ...
            └── frames/
                ├── <seq_name>/
                │   ├── 0.jpg ...
                └── ...
    """

    def __init__(self, dataset_paths):
        super().__init__(dataset_paths)
        self.base_path = dataset_paths["trackingnet_path"]
        self.sequence_list = self._list_sequences(self.base_path, ['TEST'])

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(set_name, seq_name)
                             for set_name, seq_name in self.sequence_list])

    def _construct_sequence(self, set_name, sequence_name):
        anno_path = '{}/{}/anno/{}.txt'.format(self.base_path, set_name, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/frames/{}'.format(self.base_path, set_name, sequence_name)
        frame_list = sorted(
            [f for f in os.listdir(frames_path) if f.endswith('.jpg')],
            key=lambda f: int(f[:-4]),
        )
        frames_list = [os.path.join(frames_path, f) for f in frame_list]

        return Sequence(sequence_name, frames_list, 'trackingnet', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []
        for s in set_ids:
            anno_dir = os.path.join(root, s, 'anno')
            sequences_cur_set = [
                (s, os.path.splitext(f)[0])
                for f in os.listdir(anno_dir) if f.endswith('.txt')
            ]
            sequence_list += sequences_cur_set
        return sequence_list
