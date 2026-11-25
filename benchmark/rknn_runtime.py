"""Lightweight Rockchip RKNN runtime bindings (ctypes) and host helpers for RV1126B.

Self-contained (depends only on NumPy and ``librknnrt.so`` on the target board) so
``benchmark/`` can be copied directly onto an RV1126B device without ONNX or PyTorch.
"""

import ctypes as c
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_U = c.c_uint32
_I = c.c_int32
_CTX = c.c_uint64

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 3
RKNN_FLAG_COLLECT_PERF_MASK = 8


class RknnTensorAttr(c.Structure):
    _fields_ = [
        ("index", _U),
        ("n_dims", _U),
        ("dims", _U * 16),
        ("name", c.c_char * 256),
        ("n_elems", _U),
        ("size", _U),
        ("fmt", _I),
        ("type", _I),
        ("qnt_type", _I),
        ("fl", c.c_int8),
        ("zp", _I),
        ("scale", c.c_float),
        ("w_stride", _U),
        ("size_with_stride", _U),
        ("pass_through", c.c_uint8),
        ("h_stride", _U),
    ]


class RknnInput(c.Structure):
    _fields_ = [
        ("index", _U),
        ("buf", c.c_void_p),
        ("size", _U),
        ("pass_through", c.c_uint8),
        ("type", _I),
        ("fmt", _I),
    ]


class RknnOutput(c.Structure):
    _fields_ = [
        ("want_float", c.c_uint8),
        ("is_prealloc", c.c_uint8),
        ("index", _U),
        ("buf", c.c_void_p),
        ("size", _U),
    ]


class RknnPerfDetail(c.Structure):
    _fields_ = [("perf_data", c.c_void_p), ("data_len", c.c_uint64)]


def load_rknn_lib(lib_path: str = "/usr/lib/librknnrt.so"):
    lib = c.CDLL(lib_path)
    sigs = {
        "rknn_init": [c.POINTER(_CTX), c.c_void_p, _U, _U, c.c_void_p],
        "rknn_query": [_CTX, _I, c.c_void_p, _U],
        "rknn_inputs_set": [_CTX, _U, c.POINTER(RknnInput)],
        "rknn_run": [_CTX, c.c_void_p],
        "rknn_outputs_get": [_CTX, _U, c.POINTER(RknnOutput), c.c_void_p],
        "rknn_outputs_release": [_CTX, _U, c.POINTER(RknnOutput)],
        "rknn_destroy": [_CTX],
    }
    for name, argtypes in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = _I
    return lib


def _check(ret: int, msg: str = "") -> None:
    if ret != 0:
        raise RuntimeError(f"RKNN call failed with code {ret}" + (f" ({msg})" if msg else ""))


def compute_template_geometry(tbb: np.ndarray) -> Dict[str, np.ndarray]:
    """Host-CPU box geometry weights for the lowered TaCo template graph."""
    x0, y0, x1, y1 = np.maximum(np.asarray(tbb, dtype=np.float32).reshape(4) * 128.0, 0.0)
    pixels = np.arange(128, dtype=np.float32)
    mask = ((pixels >= y0) & (pixels < y1))[:, None] & ((pixels >= x0) & (pixels < x1))[None, :]
    coverage = mask.reshape(8, 16, 8, 16).mean(axis=(1, 3), dtype=np.float32).reshape(64)
    total = float(coverage.sum())
    cond = (coverage / total) if total > 1e-6 else np.full(64, 1.0 / 64.0, dtype=np.float32)
    ctx_occ = mask.reshape(4, 32, 4, 32).mean(axis=(1, 3), dtype=np.float32).reshape(16)
    occupancy = np.concatenate([np.ones(16, dtype=np.float32), ctx_occ], axis=0)

    def _axis_weights(start: float, end: float) -> np.ndarray:
        coords = start + (np.arange(8, dtype=np.float32) + 0.5) * (end - start) / 8.0
        coords = np.clip(coords, 0.0, 7.0)
        affinity = np.maximum(1.0 - np.abs(coords[:, None] - np.arange(8, dtype=np.float32)), 0.0)
        return affinity.reshape(4, 2, 8).mean(axis=1)

    bx0, by0, bx1, by1 = np.clip(np.asarray(tbb, dtype=np.float32).reshape(4), 0.0, 1.0) * 8.0 - 0.5
    roi = np.einsum("yh,xw->hwyx", _axis_weights(by0, by1), _axis_weights(bx0, bx1)).reshape(1, 64, 16)
    return {
        "pool_weights": np.ascontiguousarray(cond.reshape(1, 64, 1), dtype=np.float32),
        "roi_weights": np.ascontiguousarray(roi, dtype=np.float32),
        "kernel_occupancy": np.ascontiguousarray(occupancy.reshape(1, 32, 1), dtype=np.float32),
    }


def _gelu_fp32(x: np.ndarray) -> np.ndarray:
    erf = np.fromiter((math.erf(float(v) * math.sqrt(0.5)) for v in x), dtype=np.float32, count=x.size)
    return x * (0.5 * (1.0 + erf))


class FilmHead:
    """Host-CPU FP32 evaluator for the partitioned FiLM head on RV1126B."""

    def __init__(self, npz_path: Path):
        with np.load(npz_path) as saved:
            self.params = {k: saved[k].astype(np.float32) for k in saved.files}

    def __call__(self, pooled: np.ndarray) -> np.ndarray:
        p = self.params
        vec = np.asarray(pooled, dtype=np.float32).reshape(-1)
        centered = vec - vec.mean()
        invstd = 1.0 / np.sqrt(np.mean(centered * centered) + 1e-5)
        normalized = centered * invstd * p["norm_weight"] + p["norm_bias"]
        h = _gelu_fp32(p["weight1"] @ normalized + p["bias1"])
        h = _gelu_fp32(p["weight2"] @ h + p["bias2"])
        projected = (p["projections"] @ h + p["biases"]).reshape(-1, 2, vec.size)
        projected[:, 0] += 1.0
        return projected.reshape(-1, 1, vec.size, 1, 1)


def parse_rknn_perf(perf_text: str) -> List[dict]:
    """Parse ``RKNN_QUERY_PERF_DETAIL`` table into per-operator rows."""
    rows = []
    for line in perf_text.splitlines():
        m = re.match(r"^(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU)\s+(.*)$", line)
        if not m:
            continue
        idx, op, dtype, target, rest = m.groups()
        parts = re.split(r"\s{2,}", rest.strip())
        try:
            ci = next(i for i, p in enumerate(parts) if re.fullmatch(r"\d+/\d+/\d+", p))
            time_us = int(parts[ci + 1].split()[0])
        except (StopIteration, ValueError, IndexError):
            continue
        rows.append(
            {
                "id": int(idx),
                "op": op,
                "dtype": dtype,
                "target": target,
                "time_us": time_us,
                "name": parts[-1].strip(),
            }
        )
    return rows


class RknnRunner:
    """Direct ``librknnrt.so`` runner for a single ``.rknn`` graph."""

    def __init__(self, rknn_path: Path, lib_path: str = "/usr/lib/librknnrt.so", collect_perf: bool = False):
        self.lib = load_rknn_lib(lib_path)
        self.ctx = _CTX()
        flags = RKNN_FLAG_COLLECT_PERF_MASK if collect_perf else 0
        _check(self.lib.rknn_init(c.byref(self.ctx), str(rknn_path).encode(), 0, flags, None), "rknn_init")
        try:
            counts = (_U * 2)()
            _check(self.lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM, c.byref(counts), c.sizeof(counts)))
            self.n_in, self.n_out = int(counts[0]), int(counts[1])
            self.in_names, self.in_elems = [], []
            self.out_names, self.out_shapes = [], []
            for i in range(self.n_in):
                attr = RknnTensorAttr()
                attr.index = i
                _check(self.lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR, c.byref(attr), c.sizeof(attr)))
                self.in_names.append(attr.name.decode())
                self.in_elems.append(int(attr.n_elems))
            for i in range(self.n_out):
                attr = RknnTensorAttr()
                attr.index = i
                _check(self.lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR, c.byref(attr), c.sizeof(attr)))
                self.out_names.append(attr.name.decode())
                self.out_shapes.append(tuple(int(attr.dims[d]) for d in range(attr.n_dims)))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = _CTX()

    def set_inputs(self, arrays: List[np.ndarray]) -> None:
        if len(arrays) != self.n_in:
            raise ValueError(f"Expected {self.n_in} inputs, got {len(arrays)}")
        self._pinned = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
        inps = (RknnInput * self.n_in)()
        for i, a in enumerate(self._pinned):
            if a.size != self.in_elems[i]:
                raise ValueError(f"Input {i} ({self.in_names[i]}) size {a.size} != expected {self.in_elems[i]}")
            fmt = 1 if a.ndim == 4 else 3  # 1 = NHWC, 3 = UNDEFINED
            inps[i] = RknnInput(i, a.ctypes.data, a.nbytes, 0, 0, fmt)
        _check(self.lib.rknn_inputs_set(self.ctx, self.n_in, inps), "rknn_inputs_set")

    def run_only(self) -> None:
        _check(self.lib.rknn_run(self.ctx, None), "rknn_run")

    def get_outputs(self) -> List[np.ndarray]:
        outs = (RknnOutput * self.n_out)()
        for i in range(self.n_out):
            outs[i].want_float = 1
            outs[i].index = i
        _check(self.lib.rknn_outputs_get(self.ctx, self.n_out, outs, None), "rknn_outputs_get")
        try:
            result = []
            for shape, desc in zip(self.out_shapes, outs):
                count = int(np.prod(shape))
                buf = np.ctypeslib.as_array(c.cast(desc.buf, c.POINTER(c.c_float)), shape=(count,))
                result.append(buf.copy().reshape(shape))
            return result
        finally:
            _check(self.lib.rknn_outputs_release(self.ctx, self.n_out, outs), "rknn_outputs_release")

    def run(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        self.set_inputs(arrays)
        self.run_only()
        return self.get_outputs()

    def query_perf_detail(self) -> str:
        perf = RknnPerfDetail()
        _check(self.lib.rknn_query(self.ctx, RKNN_QUERY_PERF_DETAIL, c.byref(perf), c.sizeof(perf)))
        return c.string_at(perf.perf_data, perf.data_len).decode(errors="replace")


class RknnHybridTemplateRunner:
    """Complete RV1126B hybrid template runner: CPU geometry + NPU backbone/kernels + CPU FP32 FiLM."""

    def __init__(self, template_dir: Path, lib_path: str = "/usr/lib/librknnrt.so"):
        self.manifest = json.loads((template_dir / "manifest.json").read_text())
        self.runner = RknnRunner(template_dir / self.manifest["rknn"], lib_path=lib_path, collect_perf=False)
        self.film_head = FilmHead(template_dir / self.manifest["film_head"])

    def close(self) -> None:
        self.runner.close()

    def run(self, z_nchw: np.ndarray, tbb: np.ndarray) -> Dict[str, np.ndarray]:
        values = {"z": np.ascontiguousarray(z_nchw.transpose(0, 2, 3, 1), dtype=np.float32), **compute_template_geometry(tbb)}
        ordered_in = [values[name] for name in self.runner.in_names]
        raw_outs = self.runner.run(ordered_in)
        caches = dict(zip(self.runner.out_names, raw_outs))
        pooled = caches.pop(self.manifest["pooled_name"])
        caches.update(zip(self.manifest["film_names"], self.film_head(pooled)))
        return caches
