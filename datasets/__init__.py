"""Tracking evaluation dataset loaders.

Usage::

    from datasets import get_dataset

    dataset_paths = {
        "got10k_path": "/data/got10k",
        "lasot_path":  "/data/lasot",
        ...
    }
    seqs = get_dataset("got10k_val", dataset_paths)

Available dataset names
-----------------------
got10k_test, got10k_val,
lasot, lasot_extension_subset,
tnl2k,
nfs, nfs_30, nfs_60, nfs_120, nfs_240,
uav, uavdt, dtb70, visdrone2018, uavdark135,
trackingnet,
vasttrack_test
"""

from datasets._core import Sequence, SequenceList, BaseDataset, load_text

from datasets.nfs import NFSDataset
from datasets.lasot import LaSOTDataset
from datasets.lasot_ext import LaSOTExtensionSubsetDataset
from datasets.got10k import GOT10KDataset
from datasets.trackingnet import TrackingNetDataset
from datasets.uav import UAVDataset
from datasets.uavdt import UAVDTDataset
from datasets.dtb70 import DTB70Dataset
from datasets.vasttrack import VastTrackDataset
from datasets.tnl2k import TNL2KDataset
from datasets.visdrone2018 import VisDrone2018Dataset
from datasets.uavdark135 import UAVDark135Dataset

# ── Registry ─────────────────────────────────────────────────────────────────
# Each entry: (DatasetClass, extra_kwargs)
_REGISTRY = {
    "nfs":                      (NFSDataset,                    {"fps": 30}),
    "nfs_30":                   (NFSDataset,                    {"fps": 30}),
    "nfs_60":                   (NFSDataset,                    {"fps": 60}),
    "nfs_120":                  (NFSDataset,                    {"fps": 120}),
    "nfs_240":                  (NFSDataset,                    {"fps": 240}),
    "lasot":                    (LaSOTDataset,                  {}),
    "lasot_extension_subset":   (LaSOTExtensionSubsetDataset,   {}),
    "got10k_test":              (GOT10KDataset,                 {"split": "test"}),
    "got10k_val":               (GOT10KDataset,                 {"split": "val"}),
    "trackingnet":              (TrackingNetDataset,            {}),
    "uav":                      (UAVDataset,                    {}),
    "uavdt":                    (UAVDTDataset,                  {}),
    "dtb70":                    (DTB70Dataset,                  {}),
    "vasttrack_test":           (VastTrackDataset,              {"split": "test"}),
    "tnl2k":                    (TNL2KDataset,                  {}),
    "visdrone2018":             (VisDrone2018Dataset,           {}),
    "uavdark135":               (UAVDark135Dataset,             {}),
}


def get_dataset(name: str, dataset_paths: dict) -> SequenceList:
    """Instantiate a dataset by name and return its SequenceList.

    Args:
        name: Dataset name, e.g. ``"got10k_val"``.  Case-insensitive.
        dataset_paths: Dict mapping path keys to filesystem paths, e.g.
            ``{"got10k_path": "/data/got10k"}``.

    Returns:
        :class:`SequenceList` of :class:`Sequence` objects.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            'Unknown dataset "{}". Available: {}'.format(name, sorted(_REGISTRY))
        )
    cls, extra = _REGISTRY[name]
    return cls(dataset_paths, **extra).get_sequence_list()


__all__ = [
    'Sequence', 'SequenceList', 'BaseDataset', 'load_text',
    'NFSDataset', 'LaSOTDataset', 'LaSOTExtensionSubsetDataset',
    'GOT10KDataset', 'TrackingNetDataset', 'UAVDataset',
    'UAVDTDataset', 'DTB70Dataset',
    'VastTrackDataset', 'TNL2KDataset', 'VisDrone2018Dataset', 'UAVDark135Dataset',
    'get_dataset',
]
