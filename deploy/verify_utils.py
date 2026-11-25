"""Shared numerical verification utilities for TaCo ONNX, RKNN, and ncnn exports.

Implements the paper's numerical acceptance gates against ONNX Runtime FP32:
  - Per-tensor gate: relative L2 <= 0.02 and max abs <= max(0.005, 0.05 * ||Y_ref||_inf)
  - Search-graph end-to-end gate: all outputs finite, per-tensor gate passed,
    identical score-peak cell (after clamped sigmoid * Hann2D), and decoded-box IoU >= 0.98.
  - Template-graph cache gate: all cache tensors finite with relative L2 <= 0.02.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_GATE = {
    "rel_l2_max": 0.02,
    "max_abs_floor": 0.005,
    "max_abs_ref_fraction": 0.05,
    "box_iou_min": 0.98,
}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_l2(actual: np.ndarray, ref: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-12))


def max_abs(actual: np.ndarray, ref: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(a - r)))


def hann2d(side: int = 16, centered: bool = True) -> np.ndarray:
    if centered:
        idx = np.arange(1, side + 1, dtype=np.float32)
        h1 = 0.5 * (1.0 - np.cos((2.0 * np.pi / (side + 1)) * idx))
    else:
        idx = np.arange(0, side // 2 + 1, dtype=np.float32)
        w = 0.5 * (1.0 + np.cos((2.0 * np.pi / (side + 2)) * idx))
        h1 = np.concatenate([w, np.flip(w[1 : side - side // 2], axis=0)], axis=0).astype(np.float32)
    return np.outer(h1, h1).astype(np.float32)


def decode_center_head(
    score_logits: np.ndarray,
    size_map: np.ndarray,
    offset_map: np.ndarray,
    window: np.ndarray,
) -> Tuple[Tuple[int, int], np.ndarray]:
    """Decode (score_logits, size, offset) into winning cell (iy, ix) and [cx, cy, w, h]."""
    score = np.asarray(score_logits, dtype=np.float64)
    prob = np.clip(1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0))), 1e-4, 1.0 - 1e-4)
    resp = prob.reshape(window.shape) * window
    h, w = resp.shape
    iy, ix = divmod(int(np.argmax(resp)), w)
    ox, oy = np.asarray(offset_map, dtype=np.float64).reshape(2, h, w)[:, iy, ix]
    sw, sh = np.asarray(size_map, dtype=np.float64).reshape(2, h, w)[:, iy, ix]
    return (iy, ix), np.array([(ix + ox) / w, (iy + oy) / h, sw, sh], dtype=np.float64)


def box_iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_lo, a_hi = a[:2] - 0.5 * a[2:], a[:2] + 0.5 * a[2:]
    b_lo, b_hi = b[:2] - 0.5 * b[2:], b[:2] + 0.5 * b[2:]
    inter = float(np.prod(np.maximum(0.0, np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo))))
    union = float(np.prod(np.maximum(a[2:], 0.0)) + np.prod(np.maximum(b[2:], 0.0)) - inter)
    return inter / max(union, 1e-12)


def gate_tensor(
    actual: np.ndarray,
    ref: np.ndarray,
    rel_l2_max: float = DEFAULT_GATE["rel_l2_max"],
    max_abs_floor: float = DEFAULT_GATE["max_abs_floor"],
    max_abs_ref_fraction: float = DEFAULT_GATE["max_abs_ref_fraction"],
) -> Dict:
    a = np.asarray(actual, dtype=np.float64).reshape(np.asarray(ref).shape)
    r = np.asarray(ref, dtype=np.float64)
    finite = bool(np.isfinite(a).all())
    rl2 = rel_l2(a, r) if finite else float("inf")
    mabs = max_abs(a, r) if finite else float("inf")
    ref_peak = float(np.max(np.abs(r))) if r.size else 0.0
    mabs_limit = max(float(max_abs_floor), float(max_abs_ref_fraction) * ref_peak)
    passed = finite and (rl2 <= rel_l2_max) and (mabs <= mabs_limit)
    return {
        "passed": passed,
        "finite": finite,
        "rel_l2": rl2,
        "max_abs": mabs,
        "max_abs_limit": mabs_limit,
        "norm_error": max(rl2 / rel_l2_max, mabs / max(mabs_limit, 1e-12)) if finite else float("inf"),
    }


def verify_search_outputs(
    actual_maps: Sequence[np.ndarray],
    ref_maps: Sequence[np.ndarray],
    window: np.ndarray,
    rel_l2_max: float = DEFAULT_GATE["rel_l2_max"],
    box_iou_min: float = DEFAULT_GATE["box_iou_min"],
) -> Dict:
    """Verify [score, size, offset] outputs against ORT FP32 reference."""
    names = ("score", "size", "offset")
    per_tensor = {}
    reshaped_actual = []
    for name, act, ref in zip(names, actual_maps, ref_maps):
        arr = np.asarray(act, dtype=np.float64).reshape(np.asarray(ref).shape)
        reshaped_actual.append(arr)
        per_tensor[name] = gate_tensor(arr, ref, rel_l2_max=rel_l2_max)

    ref_cell, ref_box = decode_center_head(ref_maps[0], ref_maps[1], ref_maps[2], window)
    act_cell, act_box = decode_center_head(reshaped_actual[0], reshaped_actual[1], reshaped_actual[2], window)
    iou = box_iou_cxcywh(act_box, ref_box)
    peak_match = (act_cell == ref_cell)
    all_tensors_ok = all(v["passed"] for v in per_tensor.values())
    passed = all_tensors_ok and peak_match and (iou >= box_iou_min)
    return {
        "passed": passed,
        "tensors": per_tensor,
        "max_rel_l2": max(v["rel_l2"] for v in per_tensor.values()),
        "max_norm_error": max(v["norm_error"] for v in per_tensor.values()),
        "ref_cell": list(ref_cell),
        "actual_cell": list(act_cell),
        "peak_match": peak_match,
        "box_iou": iou,
    }


def verify_template_caches(
    actual_caches: Dict[str, np.ndarray],
    ref_caches: Dict[str, np.ndarray],
    rel_l2_max: float = DEFAULT_GATE["rel_l2_max"],
) -> Dict:
    """Verify all cached template outputs against ORT FP32 reference."""
    errors = {}
    for name, ref in ref_caches.items():
        if name not in actual_caches:
            raise KeyError(f"Missing template cache output: {name}")
        act = np.asarray(actual_caches[name], dtype=np.float64).reshape(np.asarray(ref).shape)
        finite = bool(np.isfinite(act).all())
        rl2 = rel_l2(act, ref) if finite else float("inf")
        errors[name] = {
            "passed": finite and (rl2 <= rel_l2_max),
            "finite": finite,
            "rel_l2": rl2,
            "max_abs": max_abs(act, ref) if finite else float("inf"),
        }
    worst = max(v["rel_l2"] for v in errors.values())
    return {
        "passed": all(v["passed"] for v in errors.values()),
        "max_rel_l2": worst,
        "tensors": errors,
    }


def make_ort_session(onnx_path, threads: int = 2):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def generate_ort_references(
    template_onnx: Path,
    search_onnx: Path,
    num_samples: int = 8,
    seed: int = 0,
    template_size: int = 128,
    search_size: int = 256,
    feat_size: int = 16,
) -> Dict:
    """Create deterministic inputs and compute ONNX Runtime FP32 reference outputs.

    Uses smooth ImageNet-normalized synthetic crops and realistic centered target boxes
    so verification works out-of-the-box without requiring benchmark datasets on disk.
    """
    rng = np.random.default_rng(seed)
    tpl_sess = make_ort_session(template_onnx, threads=2)
    srch_sess = make_ort_session(search_onnx, threads=2)
    cache_names = [o.name for o in tpl_sess.get_outputs()]
    window = hann2d(feat_size, centered=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]

    def _smooth_crop(size: int) -> np.ndarray:
        # Low-frequency + medium-frequency RGB patterns normalized like real images
        base = rng.uniform(0.15, 0.85, size=(1, 3, size // 8, size // 8)).astype(np.float32)
        up = np.repeat(np.repeat(base, 8, axis=2), 8, axis=3)
        noise = rng.normal(0.0, 0.08, size=(1, 3, size, size)).astype(np.float32)
        img = np.clip(up + noise, 0.0, 1.0)
        return np.ascontiguousarray((img - mean) / std, dtype=np.float32)

    samples = []
    for idx in range(num_samples):
        z = _smooth_crop(template_size)
        x = _smooth_crop(search_size)
        bw = float(rng.uniform(0.28, 0.58))
        bh = float(rng.uniform(0.28, 0.58))
        cx = 0.5 + float(rng.uniform(-0.04, 0.04))
        cy = 0.5 + float(rng.uniform(-0.04, 0.04))
        tbb = np.array([[max(0.05, cx - bw / 2), max(0.05, cy - bh / 2),
                         min(0.95, cx + bw / 2), min(0.95, cy + bh / 2)]], dtype=np.float32)

        cache_vals = tpl_sess.run(cache_names, {"z": z, "tbb": tbb})
        caches = {n: np.ascontiguousarray(v, dtype=np.float32) for n, v in zip(cache_names, cache_vals)}
        score, size, offset = srch_sess.run(["score", "size", "offset"], {"search": x, **caches})
        samples.append(
            {
                "index": idx,
                "z": z,
                "tbb": tbb,
                "search": x,
                "caches": caches,
                "search_refs": {
                    "score": np.ascontiguousarray(score, dtype=np.float32),
                    "size": np.ascontiguousarray(size, dtype=np.float32),
                    "offset": np.ascontiguousarray(offset, dtype=np.float32),
                },
            }
        )

    return {
        "cache_names": cache_names,
        "hann": window,
        "samples": samples,
    }
