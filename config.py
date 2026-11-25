"""Model registry and benchmark dataset paths for TaCo inference.

Bundled weights are stored once per variant in ``weights/<model>.pth`` (FP16
weights with FP32 normalization parameters, ~48 MiB total for TaCo-T/S/B).
Run ``python deploy/export_onnx.py --model <model>`` to export the two-branch
``template.onnx`` and ``search.onnx`` graphs into ``onnx_models/<model>/``.

Benchmark and training data default to ``./data``. Set ``TACO_DATA_ROOT`` to
use a different parent directory.
"""

import os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# TaCo model registry (PyTorch weights + exported ONNX locations)
# ---------------------------------------------------------------------------
MODELS = {
    "TaCo-T": {
        "name": "TaCo-T",
        "weights": os.path.join(_HERE, "weights", "TaCo-T.pth"),
        "template_onnx": os.path.join(_HERE, "onnx_models", "TaCo-T", "template.onnx"),
        "search_onnx": os.path.join(_HERE, "onnx_models", "TaCo-T", "search.onnx"),
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
    },
    "TaCo-S": {
        "name": "TaCo-S",
        "weights": os.path.join(_HERE, "weights", "TaCo-S.pth"),
        "template_onnx": os.path.join(_HERE, "onnx_models", "TaCo-S", "template.onnx"),
        "search_onnx": os.path.join(_HERE, "onnx_models", "TaCo-S", "search.onnx"),
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
    },
    "TaCo-B": {
        "name": "TaCo-B",
        "weights": os.path.join(_HERE, "weights", "TaCo-B.pth"),
        "template_onnx": os.path.join(_HERE, "onnx_models", "TaCo-B", "template.onnx"),
        "search_onnx": os.path.join(_HERE, "onnx_models", "TaCo-B", "search.onnx"),
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
    },
}

# ---------------------------------------------------------------------------
# Dataset directory discovery
# ---------------------------------------------------------------------------
def _folder_key(name):
    return "".join(char for char in name.casefold() if char.isalnum())


def resolve_folder(parent, *aliases):
    """Find the first matching folder, ignoring case and punctuation."""
    parent = Path(parent).expanduser()
    if parent.is_dir():
        directories = sorted(path for path in parent.iterdir() if path.is_dir())
        for alias in aliases:
            exact = parent / alias
            if exact.is_dir():
                return exact
            match = next((path for path in directories if _folder_key(path.name) == _folder_key(alias)), None)
            if match is not None:
                return match
    return parent / aliases[0]


def default_data_root():
    return Path(os.environ.get("TACO_DATA_ROOT") or "./data").expanduser()


def resolve_data_roots(root):
    return {
        "lasot": resolve_folder(root, "LaSOT"),
        "got10k": resolve_folder(root, "GOT-10k"),
        "coco": resolve_folder(root, "COCO", "COCO2017", "COCO17"),
        "vasttrack": resolve_folder(root, "VastTrack"),
        "trackingnet": resolve_folder(root, "TrackingNet"),
    }


_data_root = default_data_root()
_roots = resolve_data_roots(_data_root)
DATASET_PATHS = {
    "lasot_path": str(_roots["lasot"]),
    "lasot_extension_subset_path": str(resolve_folder(_data_root, "LaSOT_extension_subset")),
    "got10k_path": str(_roots["got10k"]),
    "trackingnet_path": str(_roots["trackingnet"]),
    "vasttrack_test_path": str(resolve_folder(_roots["vasttrack"], "test")),
    "uav_path": str(resolve_folder(_data_root, "UAV123")),
    "uavdt_path": str(resolve_folder(_data_root, "UAVDT")),
    "dtb70_path": str(resolve_folder(_data_root, "DTB70")),
    "nfs_path": str(resolve_folder(_data_root, "NFS")),
    "tnl2k_path": str(resolve_folder(_data_root, "TNL2K", "tnl2k_test")),
    "visdrone2018_path": str(resolve_folder(_data_root, "VisDrone2018", "VisDrone2018-SOT", "VisDrone2018-SOT-test")),
    "uavdark135_path": str(resolve_folder(_data_root, "UAVDark135")),
}

# ---------------------------------------------------------------------------
# Default output directory for tracking predictions
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(_HERE, "results")
