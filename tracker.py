"""Self-contained TaCo single-object tracker supporting ONNX Runtime and PyTorch.

Both backends share identical pure-NumPy / OpenCV data loading, cropping,
preprocessing, and box decoding:
  - ``TaCoONNXTracker``:    zero PyTorch dependency (imports only numpy, cv2, onnxruntime).
  - ``TaCoPyTorchTracker``: lazy-imports PyTorch only when instantiated; supports CPU
                            and CUDA GPU (FP32 or FP16 autocast) with template caching.

Use ``create_tracker(model_cfg, backend="auto"|"pytorch"|"onnx", ...)`` to select
a backend cleanly.
"""

import math
import multiprocessing
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

BoxXYWH = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Pure-NumPy / OpenCV preprocessing and geometry (shared by ONNX & PyTorch)
# ---------------------------------------------------------------------------
class NumpyPreprocessor:
    """ImageNet normalization: (img / 255 - mean) / std, HWC uint8 -> 1CHW float32."""

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        x = img_rgb.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)


def sample_target(
    im: np.ndarray,
    target_bb: BoxXYWH,
    search_area_factor: float,
    output_sz: int,
) -> Tuple[np.ndarray, float]:
    """Extract a square crop centered at ``target_bb`` [x, y, w, h] and resize to ``output_sz``."""
    x, y, w, h = [float(v) for v in np.asarray(target_bb, dtype=np.float64).reshape(-1)]
    crop_sz = int(math.ceil(math.sqrt(max(w * h, 1e-6)) * float(search_area_factor)))
    if crop_sz < 1:
        raise ValueError(f"Bounding box is too small to crop: {target_bb}")

    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = x1 + crop_sz
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
    if x1_pad or y1_pad or x2_pad or y2_pad:
        im_crop = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    resize_factor = float(output_sz) / float(crop_sz)
    im_crop_resized = cv2.resize(im_crop, (int(output_sz), int(output_sz)))
    return im_crop_resized, resize_factor


def transform_image_to_crop(
    box_in: BoxXYWH,
    box_extract: BoxXYWH,
    resize_factor: float,
    crop_sz: float,
    normalize: bool = True,
) -> np.ndarray:
    """Transform a box [x, y, w, h] from image coordinates into crop coordinates.

    Matches TaCo's non-legacy normalization (divides by ``crop_sz``).
    """
    box_in_arr = np.asarray(box_in, dtype=np.float32).reshape(-1)
    box_ext_arr = np.asarray(box_extract, dtype=np.float32).reshape(-1)

    ext_center = box_ext_arr[0:2] + 0.5 * box_ext_arr[2:4]
    in_center = box_in_arr[0:2] + 0.5 * box_in_arr[2:4]

    out_center = (float(crop_sz) - 1.0) / 2.0 + (in_center - ext_center) * float(resize_factor)
    out_wh = box_in_arr[2:4] * float(resize_factor)
    out_xywh = np.concatenate([out_center - 0.5 * out_wh, out_wh], axis=0).astype(np.float32)

    if normalize:
        return out_xywh / float(crop_sz)
    return out_xywh


def xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in np.asarray(box_xywh, dtype=np.float32).reshape(-1)]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def clip_box(box_xywh: List[float], height: int, width: int, margin: int = 10) -> List[float]:
    x1, y1, w, h = [float(v) for v in box_xywh]
    x1 = min(max(0.0, x1), float(width - margin))
    y1 = min(max(0.0, y1), float(height - margin))
    w = min(max(float(margin), w), float(width))
    h = min(max(float(margin), h), float(height))
    return [x1, y1, w, h]


def hann1d(side: int, centered: bool = True) -> np.ndarray:
    if side <= 0:
        raise ValueError("Feature map size must be positive.")
    if centered:
        idx = np.arange(1, side + 1, dtype=np.float32)
        return 0.5 * (1.0 - np.cos((2.0 * np.pi / (side + 1)) * idx))
    idx = np.arange(0, side // 2 + 1, dtype=np.float32)
    w = 0.5 * (1.0 + np.cos((2.0 * np.pi / (side + 2)) * idx))
    right = np.flip(w[1 : side - side // 2], axis=0)
    return np.concatenate([w, right], axis=0).astype(np.float32)


def hann2d(side: int, centered: bool = True) -> np.ndarray:
    h1 = hann1d(side, centered=centered)
    return np.outer(h1, h1).astype(np.float32)


def clamped_sigmoid(logits: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -50.0, 50.0)))
    return np.clip(prob, eps, 1.0 - eps).astype(np.float32)


def decode_center_head(
    score_logits: np.ndarray,
    size_map: np.ndarray,
    offset_map: np.ndarray,
    window: np.ndarray,
) -> Tuple[List[float], float]:
    """Decode raw center-head outputs into normalized [cx, cy, w, h] and peak score."""
    score_prob = clamped_sigmoid(score_logits)[0, 0]  # (H, W)
    response = score_prob * window
    feat_h, feat_w = response.shape
    idx = int(np.argmax(response))
    iy, ix = divmod(idx, feat_w)

    ox = float(offset_map[0, 0, iy, ix])
    oy = float(offset_map[0, 1, iy, ix])
    sw, sh = map(float, np.clip(size_map[0, :, iy, ix], 1e-4, 1.0 - 1e-4))

    cx = (ix + ox) / float(feat_w)
    cy = (iy + oy) / float(feat_h)
    return [cx, cy, sw, sh], float(response[iy, ix])


class _BaseTaCoTracker:
    """Shared NumPy preprocessing, box mapping, and post-processing logic."""

    def __init__(self, model_cfg: dict):
        self.template_size = int(model_cfg["template_size"])
        self.search_size = int(model_cfg["search_size"])
        self.template_factor = float(model_cfg["template_factor"])
        self.search_factor = float(model_cfg["search_factor"])
        self.stride = int(model_cfg["stride"])
        self.feat_sz = self.search_size // self.stride
        self.output_window = hann2d(self.feat_sz, centered=True)
        self.preprocessor = NumpyPreprocessor()
        self.state: Optional[List[float]] = None
        self.frame_id: int = 0

    def _prepare_template(self, image: np.ndarray, init_bbox: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        bbox = [float(v) for v in init_bbox]
        z_patch, resize_factor = sample_target(image, bbox, self.template_factor, self.template_size)
        z_tensor = self.preprocessor.process(z_patch)
        tb_xywh = transform_image_to_crop(bbox, bbox, resize_factor, float(self.template_size), normalize=True)
        tbb_xyxy = xywh_to_xyxy(tb_xywh)[None, :]  # (1, 4) float32
        self.state = bbox
        self.frame_id = 0
        return z_tensor, tbb_xyxy

    def _prepare_search(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.state is None:
            raise RuntimeError("Tracker must be initialized with initialize() before calling track().")
        self.frame_id += 1
        x_patch, resize_factor = sample_target(image, self.state, self.search_factor, self.search_size)
        x_tensor = self.preprocessor.process(x_patch)
        return x_tensor, resize_factor

    def _postprocess(
        self,
        score_logits: np.ndarray,
        size_map: np.ndarray,
        offset_map: np.ndarray,
        resize_factor: float,
        height: int,
        width: int,
    ) -> dict:
        box_cxcywh, best_score = decode_center_head(score_logits, size_map, offset_map, self.output_window)
        scale = float(self.search_size) / float(resize_factor)
        pred_box = [v * scale for v in box_cxcywh]
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), height, width, margin=10)
        return {"target_bbox": self.state, "best_score": best_score}

    def _map_box_back(self, pred_box: List[float], resize_factor: float) -> List[float]:
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * float(self.search_size) / float(resize_factor)
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


# ---------------------------------------------------------------------------
# Backend 1: PyTorch-free ONNX Runtime Tracker
# ---------------------------------------------------------------------------
def _select_ort_providers(provider: str, device_id: int) -> Tuple[List[str], List[dict]]:
    import onnxruntime as ort

    avail = set(ort.get_available_providers())
    if provider == "cpu":
        return ["CPUExecutionProvider"], [{}]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(f"CUDAExecutionProvider requested but available providers are: {sorted(avail)}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": int(device_id)}, {}]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": int(device_id)}, {}]
    return ["CPUExecutionProvider"], [{}]


class TaCoONNXTracker(_BaseTaCoTracker):
    """Single-object tracker powered solely by ONNX Runtime, NumPy, and OpenCV (no PyTorch)."""

    def __init__(
        self,
        model_cfg: dict,
        device_id: int = 0,
        provider: str = "auto",
        num_threads: Optional[int] = None,
    ):
        super().__init__(model_cfg)
        import onnxruntime as ort

        tpl_path = model_cfg["template_onnx"]
        srch_path = model_cfg["search_onnx"]
        if not os.path.isfile(tpl_path) or not os.path.isfile(srch_path):
            name = model_cfg.get("name", "TaCo-T")
            raise FileNotFoundError(
                f"ONNX files not found ({tpl_path}, {srch_path}). "
                f"Export them first with: python deploy/export_onnx.py --model {name}"
            )

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        if num_threads is not None and num_threads > 0:
            sess_options.intra_op_num_threads = int(num_threads)
            sess_options.inter_op_num_threads = 1

        providers, provider_options = _select_ort_providers(provider, device_id)
        self.template_session = ort.InferenceSession(
            tpl_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )
        self.search_session = ort.InferenceSession(
            srch_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )

        self.cache_names = [o.name for o in self.template_session.get_outputs()]
        self.search_input_names = [i.name for i in self.search_session.get_inputs()]
        self.search_output_names = [o.name for o in self.search_session.get_outputs()]

        if self.search_input_names != ["search"] + self.cache_names:
            raise RuntimeError(
                f"Template outputs {self.cache_names} do not match search inputs {self.search_input_names}"
            )
        self.cache: Optional[Dict[str, np.ndarray]] = None

        if multiprocessing.current_process().name == "MainProcess":
            active_ep = self.search_session.get_providers()[0]
            print(
                f"[TaCo/ONNX] provider={active_ep}, caches={len(self.cache_names)}, "
                f"template={self.template_size}, search={self.search_size}"
            )

    def initialize(self, image: np.ndarray, info: dict) -> None:
        z_tensor, tbb_xyxy = self._prepare_template(image, info["init_bbox"])
        cache_values = self.template_session.run(self.cache_names, {"z": z_tensor, "tbb": tbb_xyxy})
        self.cache = {
            name: np.ascontiguousarray(val, dtype=np.float32)
            for name, val in zip(self.cache_names, cache_values)
        }

    def track(self, image: np.ndarray) -> dict:
        if self.cache is None:
            raise RuntimeError("Tracker must be initialized with initialize() before calling track().")
        height, width = image.shape[:2]
        x_tensor, resize_factor = self._prepare_search(image)
        score_logits, size_map, offset_map = self.search_session.run(
            self.search_output_names, {"search": x_tensor, **self.cache}
        )
        return self._postprocess(score_logits, size_map, offset_map, resize_factor, height, width)


# ---------------------------------------------------------------------------
# Backend 2: PyTorch Tracker (CPU & CUDA GPU, NumPy preprocessing)
# ---------------------------------------------------------------------------
class TaCoPyTorchTracker(_BaseTaCoTracker):
    """TaCo tracker using PyTorch for network inference and NumPy/OpenCV for preprocessing.

    Supports both CPU (FP32) and CUDA GPU (FP32 or FP16 autocast via ``fp16=True``).
    PyTorch is imported lazily inside ``__init__`` so ONNX-only users never need PyTorch.
    """

    def __init__(
        self,
        model_cfg: dict,
        device_id: int = 0,
        device: str = "auto",
        fp16: bool = False,
        num_threads: Optional[int] = None,
    ):
        super().__init__(model_cfg)
        import torch
        from models import build_taco

        self._torch = torch
        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(int(num_threads))

        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested for PyTorch backend, but torch.cuda.is_available() is False.")
            self.device = torch.device(f"cuda:{int(device_id)}")
        else:  # auto
            self.device = torch.device(f"cuda:{int(device_id)}") if torch.cuda.is_available() else torch.device("cpu")

        self.use_fp16 = bool(fp16) and self.device.type == "cuda"

        weights_path = model_cfg["weights"]
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"PyTorch weights not found: {weights_path}")

        variant_name = model_cfg["name"]
        self.model = build_taco(variant_name, deploy=True, weights=weights_path).float().to(self.device).eval()
        self.cache = None

        if multiprocessing.current_process().name == "MainProcess":
            prec = "FP16-AMP" if self.use_fp16 else "FP32"
            print(
                f"[TaCo/PyTorch] model={variant_name}, device={self.device}, precision={prec}, "
                f"template={self.template_size}, search={self.search_size}"
            )

    def initialize(self, image: np.ndarray, info: dict) -> None:
        torch = self._torch
        z_np, tbb_np = self._prepare_template(image, info["init_bbox"])
        z_t = torch.from_numpy(z_np).to(self.device, non_blocking=True)
        tbb_t = torch.from_numpy(tbb_np).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    self.cache = self.model.forward_template(z_t, tbb_t)
            else:
                self.cache = self.model.forward_template(z_t, tbb_t)

    def track(self, image: np.ndarray) -> dict:
        if self.cache is None:
            raise RuntimeError("Tracker must be initialized with initialize() before calling track().")
        torch = self._torch
        height, width = image.shape[:2]
        x_np, resize_factor = self._prepare_search(image)
        x_t = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.forward_search(x_t, self.cache, return_logits=True)
            else:
                out = self.model.forward_search(x_t, self.cache, return_logits=True)
        score_logits = out["score_logits"].float().cpu().numpy()
        size_map = out["size_map"].float().cpu().numpy()
        offset_map = out["offset_map"].float().cpu().numpy()
        return self._postprocess(score_logits, size_map, offset_map, resize_factor, height, width)


def create_tracker(
    model_cfg: dict,
    backend: str = "auto",
    device_id: int = 0,
    provider: str = "auto",
    fp16: bool = False,
    num_threads: Optional[int] = None,
    weights: Optional[str] = None,
) -> _BaseTaCoTracker:
    """Instantiate either ``TaCoPyTorchTracker`` or ``TaCoONNXTracker``.

    When ``backend == "auto"``, selects ONNX if both exported ONNX graphs exist,
    otherwise selects PyTorch with the bundled ``weights/<model>.pth``.
    Supplying explicit PyTorch weights selects the PyTorch backend.
    """
    if weights is not None:
        if backend == "onnx":
            raise ValueError("PyTorch weights require --backend pytorch or auto.")
        model_cfg = dict(model_cfg, weights=os.path.expanduser(weights))
        backend = "pytorch"
    if backend == "auto":
        has_onnx = os.path.isfile(model_cfg["template_onnx"]) and os.path.isfile(model_cfg["search_onnx"])
        backend = "onnx" if has_onnx else "pytorch"

    if backend == "onnx":
        return TaCoONNXTracker(model_cfg, device_id=device_id, provider=provider, num_threads=num_threads)
    if backend == "pytorch":
        return TaCoPyTorchTracker(
            model_cfg, device_id=device_id, device=provider, fp16=fp16, num_threads=num_threads
        )
    raise ValueError(f"Unsupported backend {backend!r}; choose from 'auto', 'pytorch', 'onnx'.")
