#!/usr/bin/env python3
"""Report the runtime and data available to TaCo."""

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import default_data_root, resolve_data_roots, resolve_folder


def module_status(name):
    try:
        module = importlib.import_module(name)
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"{name}: unavailable ({exc})")
        return None
    print(f"{name}: {getattr(module, '__version__', 'available')}")
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    torch = module_status("torch")
    if torch is not None:
        cuda = torch.cuda.is_available()
        print(f"CUDA: {'available' if cuda else 'unavailable'} (build={torch.version.cuda})")
        for index in range(torch.cuda.device_count()):
            print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
        torchvision = module_status("torchvision")
        if torchvision is not None:
            devices = ["cpu", "cuda"] if cuda else ["cpu"]
            for device in devices:
                try:
                    image = torch.zeros(1, 1, 4, 4, device=device)
                    rois = torch.tensor([[0, 0, 0, 3, 3]], dtype=torch.float32, device=device)
                    torchvision.ops.roi_align(image, rois, (2, 2), aligned=True).sum().item()
                    print(f"RoIAlign ({device}): works")
                except (RuntimeError, NotImplementedError, ImportError) as exc:
                    print(f"RoIAlign ({device}): unavailable ({exc})")
    module_status("cv2")
    module_status("yaml")
    module_status("pycocotools")
    module_status("nvidia.dali")
    try:
        from train_lib.data.image_loader import jpeg4py_available

        print(f"jpeg4py/libturbojpeg: {'available' if jpeg4py_available() else 'OpenCV fallback'}")
    except ImportError as exc:
        print(f"Image loader: unavailable ({exc})")

    roots = resolve_data_roots(args.data_root)
    roots["got10k"] = roots["got10k"] if (roots["got10k"] / "list.txt").is_file() else resolve_folder(
        roots["got10k"], "train"
    )
    for name, path in roots.items():
        print(f"{name}: {path} ({'found' if path.is_dir() else 'missing'})")
    for name in ("lowformer_E1.pt", "repvit_m1_1_distill_300e.pth", "repvit_m1_5_distill_300e.pth"):
        path = REPO_ROOT / "pretrained_models" / name
        print(f"{name}: {'found' if path.is_file() else 'missing'}")


if __name__ == "__main__":
    main()
