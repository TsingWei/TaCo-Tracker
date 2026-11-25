"""Load RGB images with an optional JPEG accelerator."""

from functools import lru_cache
from pathlib import Path

import cv2


@lru_cache(maxsize=1)
def jpeg4py_available():
    try:
        import jpeg4py
        # Importing the wrapper alone does not load the native library.
        jpeg4py._cffi.initialize()
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False
    return True


def jpeg4py_loader(path):
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg"} and jpeg4py_available():
        import jpeg4py

        try:
            return jpeg4py.JPEG(str(path)).decode()
        except (OSError, RuntimeError, ValueError):
            pass
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Unable to decode image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
