"""Template graph lowering and RV1126B FiLM-head partitioning for TaCo.

Edge runtimes (ncnn and RKNN) do not natively support dynamic box-dependent
``RoiAlign`` or mixed integer/float box-mask indexing inside static graphs.
This module performs two exact, weight-preserving ONNX graph transformations:

1. ``lower_template_onnx`` (used for both ncnn and RKNN):
   - Moves the lightweight box-geometry weight computation (``pool_weights``,
     ``roi_weights``, ``kernel_occupancy``) to the host CPU (``compute_template_geometry``).
   - Replaces ``Einsum`` (coverage-weighted box pooling) and ``RoiAlign`` (4x4
     bilinear foreground kernel sampling) with exact ``MatMul`` operations.

2. ``split_rknn_template_onnx`` (used for RV1126B NPU):
   - Partitions the FiLM conditioning MLP (``LayerNorm -> Linear -> GELU -> Linear
     -> GELU -> per-site projections``) into ``film_head.npz`` executed in FP32 on
     the host CPU (avoiding FP16 drift on the RV1126B NPU while keeping the entire
     template backbone and kernel generation on the NPU).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def compute_template_geometry(tbb: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute exact host-side box geometry operands from normalized xyxy ``tbb`` (1, 4).

    Returns:
      - ``pool_weights``:     (1, 64, 1)  coverage-weighted box pooling weights on the 8x8 map
      - ``roi_weights``:      (1, 64, 16) bilinear 4x4 RoIAlign sampling matrix (sampling_ratio=2, aligned=True)
      - ``kernel_occupancy``: (1, 32, 1)  fractional in-box coverage a_i (1.0 for 16 target kernels,
                                          box-cell overlap for 16 context kernels)
    """
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
        return affinity.reshape(4, 2, 8).mean(axis=1)  # (4, 8)

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
    """Host-CPU FP32 evaluator for the partitioned FiLM conditioning MLP."""

    def __init__(self, npz_path):
        with np.load(npz_path) as saved:
            self.params = {name: saved[name].astype(np.float32) for name in saved.files}

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


def lower_template_onnx(src_onnx: Path, dst_onnx: Path, broadcast_pool: bool = False) -> Path:
    """Lower ``template.onnx`` by replacing box-dependent ops with MatMul inputs.

    Produces a static graph taking ``(z, pool_weights, roi_weights, kernel_occupancy)``
    with identical outputs (relative L2 < 1e-4 against the original ONNX graph).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src_onnx))
    graph = model.graph
    nodes = list(graph.node)
    outputs = {v.name for v in graph.output}

    einsum_nodes = [n for n in nodes if n.op_type == "Einsum"]
    roi_nodes = [n for n in nodes if n.op_type == "RoiAlign"]
    if len(einsum_nodes) != 1 or len(roi_nodes) != 1:
        raise RuntimeError(
            f"Expected 1 Einsum and 1 RoiAlign node in template graph, got {len(einsum_nodes)} and {len(roi_nodes)}"
        )
    einsum = einsum_nodes[0]
    roi = roi_nodes[0]

    pwc_out = next(v for v in graph.output if v.name == "pwc_s3")
    channels = int(pwc_out.type.tensor_type.shape.dim[-1].dim_value)

    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1, channels, 64], dtype=np.int64), "template_flat_shape"),
            numpy_helper.from_array(np.array([1, channels, 4, 4], dtype=np.int64), "template_roi_shape"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "template_pool_axis"),
        ]
    )

    # Locate the shared stride-16 feature tensor feeding both Einsum (via Reshape) and RoiAlign (via Cast or directly)
    reshape = next(n for n in nodes if einsum.input[1] in n.output)
    roi_producer = next((n for n in nodes if roi.input[0] in n.output), None)
    feat_tensor = reshape.input[0]
    if roi_producer is not None and roi_producer.op_type == "Cast":
        assert roi_producer.input[0] == feat_tensor, (roi_producer.input[0], feat_tensor)

    # Locate the Concat node that appends in-box occupancy a_i to kf_norm output before kf_mlp
    kf_concat = None
    for n in nodes:
        if n.op_type == "Concat" and any("kf_norm" in inp for inp in n.input):
            kf_concat = n
            break
    if kf_concat is None:
        raise RuntimeError("Could not locate kernel-occupancy Concat node in template.onnx")
    kf_concat.input[1] = "kernel_occupancy"

    pool_nodes = [
        helper.make_node("Reshape", [feat_tensor, "template_flat_shape"], ["template_flat"]),
        helper.make_node("MatMul", ["template_flat", "pool_weights"], ["template_pool"]),
        helper.make_node("Squeeze", ["template_pool", "template_pool_axis"], list(einsum.output)),
    ]
    if broadcast_pool:
        graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([1, 1, 8, 8], dtype=np.int64), "template_mask_shape"),
                numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "template_spatial_axes"),
            ]
        )
        pool_nodes = [
            helper.make_node("Reshape", [feat_tensor, "template_flat_shape"], ["template_flat"]),
            helper.make_node("Reshape", ["pool_weights", "template_mask_shape"], ["template_mask"]),
            helper.make_node("Mul", [feat_tensor, "template_mask"], ["template_weighted"]),
            helper.make_node("ReduceSum", ["template_weighted", "template_spatial_axes"], list(einsum.output), keepdims=0),
        ]

    replacements = {
        einsum.name: pool_nodes,
        roi.name: [
            helper.make_node("MatMul", ["template_flat", "roi_weights"], ["template_roi_flat"]),
            helper.make_node("Reshape", ["template_roi_flat", "template_roi_shape"], list(roi.output)),
        ],
    }

    changed = [new for n in nodes for new in replacements.get(n.name, [n])]
    keep = []
    needed = set(outputs)
    for n in reversed(changed):
        if needed.intersection(n.output):
            keep.append(n)
            needed.update(n.input)
    del graph.node[:]
    graph.node.extend(reversed(keep))

    for i in reversed(range(len(graph.initializer))):
        if graph.initializer[i].name not in needed:
            del graph.initializer[i]

    assert "tbb" not in needed, "tbb should be completely eliminated after lowering"
    del graph.input[1:]  # keep only 'z'
    for name, shape in (
        ("pool_weights", [1, 64, 1]),
        ("roi_weights", [1, 64, 16]),
        ("kernel_occupancy", [1, 32, 1]),
    ):
        assert name in needed, f"Lowered input {name} is unused"
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

    onnx.checker.check_model(model)
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_onnx))
    return dst_onnx


def split_rknn_template_onnx(lowered_onnx: Path, dst_dir: Path) -> Tuple[Path, Path, Dict]:
    """Partition the FiLM MLP out of a lowered template ONNX for hybrid RV1126B execution.

    Writes:
      - ``dst_dir / 'template_npu.onnx'``: backbone + kernel generation + pooled descriptor output
      - ``dst_dir / 'film_head.npz'``:     FP32 weights for CPU ``FilmHead`` evaluation
    Returns ``(npu_onnx_path, film_npz_path, metadata_dict)``.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    dst_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(lowered_onnx))
    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    norm = next(
        n for n in graph.node if n.op_type == "LayerNormalization" and "cond" in n.name
    )
    out_names = [o.name for o in graph.output]
    film_count = sum(1 for n in out_names if n.startswith("film_s3_")) // 2
    film_names = [name for i in range(film_count) for name in (f"film_s3_{i}_scale", f"film_s3_{i}_shift")]
    assert set(out_names[: 2 * film_count]) == set(film_names), (out_names[:4], film_names[:4])

    # Match initializer prefix ('bb.cond.1.weight' or 'cond.1.weight')
    cond1_w = next(k for k in init if k.endswith("cond.1.weight"))
    prefix = cond1_w[: -len("cond.1.weight")]
    proj_prefix = [f"{prefix}film.s3.{i}.proj" for i in range(film_count)]

    film_npz = dst_dir / "film_head.npz"
    np.savez(
        film_npz,
        norm_weight=init[norm.input[1]],
        norm_bias=init[norm.input[2]],
        weight1=init[f"{prefix}cond.1.weight"],
        bias1=init[f"{prefix}cond.1.bias"],
        weight2=init[f"{prefix}cond.3.weight"],
        bias2=init[f"{prefix}cond.3.bias"],
        projections=np.concatenate([init[p + ".weight"] for p in proj_prefix], axis=0),
        biases=np.concatenate([init[p + ".bias"] for p in proj_prefix], axis=0),
    )

    kept_outputs = [o for o in graph.output if o.name not in film_names]
    pooled_name = norm.input[0]
    channels = int(init[norm.input[1]].size)
    kept_outputs.append(helper.make_tensor_value_info(pooled_name, TensorProto.FLOAT, [1, channels]))

    needed = {o.name for o in kept_outputs}
    keep_nodes = []
    for n in reversed(graph.node):
        if needed.intersection(n.output):
            keep_nodes.append(n)
            needed.update(n.input)
    del graph.node[:]
    graph.node.extend(reversed(keep_nodes))
    del graph.output[:]
    graph.output.extend(kept_outputs)
    for i in reversed(range(len(graph.initializer))):
        if graph.initializer[i].name not in needed:
            del graph.initializer[i]

    onnx.checker.check_model(model)
    npu_onnx = dst_dir / "template_npu.onnx"
    onnx.save(model, str(npu_onnx))

    meta = {
        "cache_names": out_names,
        "film_names": film_names,
        "pooled_name": pooled_name,
        "rk_outputs": [o.name for o in kept_outputs],
        "template_npu_sha256": hashlib.sha256(npu_onnx.read_bytes()).hexdigest(),
        "film_head_sha256": hashlib.sha256(film_npz.read_bytes()).hexdigest(),
    }
    return npu_onnx, film_npz, meta
