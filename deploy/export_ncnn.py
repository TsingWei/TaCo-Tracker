#!/usr/bin/env python3
"""Convert TaCo ONNX graphs to ncnn (via PNNX) with numerical verification.

Requires ``pnnx``, ``ncnn``, ``onnx``, and ``onnxruntime`` (tested with pnnx 20261020
and ncnn 1.0.20261020).

Conversion policy (matching the paper):
  - Runs ``pnnx <model.onnx> fp16=0`` so weights are stored in FP32 inside
    ``model.ncnn.bin``; FP16 storage/packed/arithmetic is selected at load time by ncnn.
  - For ``template``, lowers ``template.onnx`` first (moving box geometry to CPU and
    lowering ``RoiAlign`` to ``MatMul``) so every layer maps natively to ncnn.
  - Parses ``pnnx`` batch-axis decisions from ``model_ncnn.py`` into ``io_map.json``
    and checks that every layer is supported by ``ncnn``.
  - Validates converted ncnn outputs against ONNX Runtime FP32 references.

Examples:
    python deploy/export_ncnn.py --model TaCo-T --out build/ncnn/TaCo-T --graph both
    python deploy/export_ncnn.py --model TaCo-B --out build/ncnn/TaCo-B --graph search --check-fp16
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from template_lowering import compute_template_geometry, lower_template_onnx
from verify_utils import (
    file_sha256,
    generate_ort_references,
    make_ort_session,
    verify_search_outputs,
    verify_template_caches,
)


def _find_pnnx(pnnx_override: Optional[str] = None) -> str:
    if pnnx_override:
        return pnnx_override
    found = shutil.which("pnnx")
    if found:
        return found
    bin_candidate = Path(sys.executable).resolve().parent / "pnnx"
    if bin_candidate.is_file():
        return str(bin_candidate)
    import site
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        cand = Path(sp) / "pnnx" / "pnnx"
        if cand.is_file():
            return str(cand)
    try:
        import pnnx as pnnx_pkg

        if getattr(pnnx_pkg, "__file__", None):
            candidate = Path(pnnx_pkg.__file__).resolve().parent / "pnnx"
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass
    raise FileNotFoundError("pnnx binary not found in PATH or Python environment. Pass --pnnx /path/to/pnnx.")


def _parse_pnnx_batch_index(work_dir: Path) -> Dict[str, int]:
    py_file = work_dir / "model_ncnn.py"
    if not py_file.is_file():
        return {}
    text = py_file.read_text(errors="replace")
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r'ex\.input\("(in\d+)",\s*ncnn\.Mat\(in\d+\.numpy\(\),\s*batch_index=(\d+)\)', text)
    }


def _inspect_ncnn_param(param_path: Path) -> Tuple[dict, set]:
    import ncnn

    lines = param_path.read_text().splitlines()
    layers: Dict[str, int] = {}
    blobs = set()
    inputs: List[str] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        ltype, name, nin, nout = parts[0], parts[1], int(parts[2]), int(parts[3])
        layers[ltype] = layers.get(ltype, 0) + 1
        blobs.update(parts[4 : 4 + nin])
        blobs.update(parts[4 + nin : 4 + nin + nout])
        if ltype == "Input":
            inputs.append(name)
    unsupported = sorted(t for t in layers if ncnn.layer_to_index(t) < 0)
    return {
        "layer_count": sum(layers.values()),
        "layer_types": dict(sorted(layers.items(), key=lambda kv: -kv[1])),
        "unsupported_layer_types": unsupported,
        "inputs": inputs,
    }, blobs


def _build_io_map(
    onnx_path: Path,
    work_dir: Path,
    sample_inputs: Optional[Dict[str, np.ndarray]] = None,
    sample_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> dict:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    gin = [i for i in m.graph.input if i.name not in inits]
    bidx = _parse_pnnx_batch_index(work_dir)

    io_map = {"inputs": [], "outputs": []}
    for k, inp in enumerate(gin):
        if sample_inputs is not None and inp.name in sample_inputs:
            dims = list(sample_inputs[inp.name].shape)
        else:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        b = bidx.get(f"in{k}")
        if b == 233:
            ncnn_shape = list(dims)
        elif dims and dims[0] == 1:
            ncnn_shape = list(dims[1:])
        else:
            ncnn_shape = list(dims)
        io_map["inputs"].append(
            {
                "onnx": inp.name,
                "ncnn": f"in{k}",
                "onnx_shape": list(dims),
                "ncnn_shape": ncnn_shape,
                "pnnx_batch_index": b,
            }
        )
    for k, out in enumerate(m.graph.output):
        if sample_outputs is not None and out.name in sample_outputs:
            dims = list(sample_outputs[out.name].shape)
        else:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        io_map["outputs"].append(
            {
                "onnx": out.name,
                "ncnn": f"out{k}",
                "onnx_shape": list(dims),
                "onnx_dtype": "float32",
            }
        )
    return io_map


def _run_ncnn_once(net, io_map: dict, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
    import ncnn

    ex = net.create_extractor()
    for item in io_map["inputs"]:
        arr = np.ascontiguousarray(feeds[item["onnx"]], dtype=np.float32).reshape(item["ncnn_shape"])
        ex.input(item["ncnn"], ncnn.Mat(arr))
    outs = []
    for item in io_map["outputs"]:
        ret, mat = ex.extract(item["ncnn"])
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed on {item['ncnn']} (rc={ret})")
        outs.append(np.array(mat, dtype=np.float32).reshape(item["onnx_shape"]))
    return outs


def _load_ncnn_net(work_dir: Path, fp16: bool, threads: int = 4):
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = int(threads)
    net.opt.use_fp16_packed = bool(fp16)
    net.opt.use_fp16_storage = bool(fp16)
    net.opt.use_fp16_arithmetic = bool(fp16)
    net.opt.use_packing_layout = True
    net.load_param(str(work_dir / "model.ncnn.param"))
    net.load_model(str(work_dir / "model.ncnn.bin"))
    return net


def _run_pnnx_and_verify(
    onnx_path: Path,
    work_dir: Path,
    pnnx_bin: str,
    graph_kind: str,
    cases: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    window: Optional[np.ndarray],
    check_fp16: bool,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    local_onnx = work_dir / "model.onnx"
    if local_onnx.exists() or local_onnx.is_symlink():
        local_onnx.unlink()
    shutil.copy2(onnx_path, local_onnx)

    log_path = work_dir / "pnnx.log"
    with log_path.open("w") as log:
        subprocess.run([pnnx_bin, "model.onnx", "fp16=0"], cwd=str(work_dir), stdout=log, stderr=subprocess.STDOUT, check=True)
    if "FATAL ERROR" in log_path.read_text(errors="replace"):
        raise RuntimeError(f"pnnx reported FATAL ERROR; inspect {log_path}")

    io_map = _build_io_map(local_onnx, work_dir, sample_inputs=cases[0][0], sample_outputs=cases[0][1])
    summary, blobs = _inspect_ncnn_param(work_dir / "model.ncnn.param")
    if summary["unsupported_layer_types"]:
        raise RuntimeError(f"Unsupported ncnn layers: {summary['unsupported_layer_types']}")
    missing_in = [i["ncnn"] for i in io_map["inputs"] if i["ncnn"] not in summary["inputs"]]
    missing_out = [o["ncnn"] for o in io_map["outputs"] if o["ncnn"] not in blobs]
    if missing_in or missing_out:
        raise RuntimeError(f"Missing ncnn blobs: inputs={missing_in}, outputs={missing_out}")
    (work_dir / "io_map.json").write_text(json.dumps(io_map, indent=2) + "\n")

    # Save refs.npz for downstream Pi 5 benchmark verification
    ref_arrays = {}
    if window is not None:
        ref_arrays["__hann__"] = window
    for idx, (feed, expected) in enumerate(cases):
        for k, v in feed.items():
            ref_arrays[f"case{idx}__in__{k}"] = np.ascontiguousarray(v, dtype=np.float32)
        for k, v in expected.items():
            ref_arrays[f"case{idx}__out__{k}"] = np.ascontiguousarray(v, dtype=np.float32)
    meta = {
        "graph_kind": graph_kind,
        "num_cases": len(cases),
        "input_names": [i["onnx"] for i in io_map["inputs"]],
        "output_names": [o["onnx"] for o in io_map["outputs"]],
        "source_onnx_sha256": file_sha256(onnx_path),
    }
    np.savez(work_dir / "refs.npz", __meta__=json.dumps(meta), **ref_arrays)

    # Numerical check in FP32 (and optionally FP16)
    checks = {}
    precisions = [("fp32", False)] + ([("fp16", True)] if check_fp16 else [])
    for label, use_fp16 in precisions:
        net = _load_ncnn_net(work_dir, fp16=use_fp16, threads=4)
        rows = []
        for feed, expected in cases:
            got = _run_ncnn_once(net, io_map, feed)
            if graph_kind == "search":
                ref_maps = [expected["score"], expected["size"], expected["offset"]]
                rows.append(verify_search_outputs(got, ref_maps, window))
            else:
                got_dict = dict(zip([o["onnx"] for o in io_map["outputs"]], got))
                rows.append(verify_template_caches(got_dict, expected))
        passed = all(r["passed"] for r in rows)
        worst_l2 = max(r["max_rel_l2"] for r in rows)
        checks[label] = {"passed": passed, "max_rel_l2": worst_l2, "num_cases": len(rows)}
        print(f"[ncnn {graph_kind} {label}] passed={passed} max_rel_l2={worst_l2:.3e} layers={summary['layer_count']}")
        if label == "fp32" and not passed:
            raise RuntimeError(f"ncnn FP32 conversion check failed for {graph_kind}: max_rel_l2={worst_l2}")

    # Clean up redundant local ONNX copy (keep param, bin, io_map, refs, manifest)
    local_onnx.unlink()

    manifest = {
        "graph_kind": graph_kind,
        "source_onnx_sha256": file_sha256(onnx_path),
        "param_sha256": file_sha256(work_dir / "model.ncnn.param"),
        "bin_sha256": file_sha256(work_dir / "model.ncnn.bin"),
        "io_map_sha256": file_sha256(work_dir / "io_map.json"),
        "refs_sha256": file_sha256(work_dir / "refs.npz"),
        "layers": summary,
        "verification": checks,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TaCo ONNX graphs to ncnn via PNNX with numerical checks.")
    parser.add_argument("--model", type=str, default="TaCo-T", choices=["TaCo-T", "TaCo-S", "TaCo-B"])
    parser.add_argument("--onnx-dir", type=Path, default=None, help="Directory containing template.onnx and search.onnx.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for ncnn models and verification refs.")
    parser.add_argument("--graph", type=str, default="both", choices=["search", "template", "both"])
    parser.add_argument("--pnnx", type=str, default=None, help="Optional explicit path to the pnnx binary.")
    parser.add_argument("--samples", type=int, default=8, help="Number of verification samples (default: 8).")
    parser.add_argument("--check-fp16", action="store_true", help="Also verify ncnn FP16 on the conversion host.")
    args = parser.parse_args()

    onnx_dir = args.onnx_dir or (REPO_ROOT / "onnx_models" / args.model)
    tpl_onnx = onnx_dir / "template.onnx"
    srch_onnx = onnx_dir / "search.onnx"
    if not tpl_onnx.is_file() or not srch_onnx.is_file():
        raise FileNotFoundError(f"Expected template.onnx and search.onnx inside {onnx_dir}")

    pnnx_bin = _find_pnnx(args.pnnx)
    ref_bundle = generate_ort_references(tpl_onnx, srch_onnx, num_samples=args.samples, seed=0)

    if args.graph in ("search", "both"):
        search_cases = [
            ({"search": s["search"], **s["caches"]}, s["search_refs"])
            for s in ref_bundle["samples"]
        ]
        _run_pnnx_and_verify(
            srch_onnx, args.out / "search", pnnx_bin, "search", search_cases, ref_bundle["hann"], args.check_fp16
        )

    if args.graph in ("template", "both"):
        tpl_dir = args.out / "template"
        tpl_dir.mkdir(parents=True, exist_ok=True)
        lowered_onnx = tpl_dir / "template_lowered.onnx"
        lower_template_onnx(tpl_onnx, lowered_onnx, broadcast_pool=False)

        # Confirm lowered graph matches original template.onnx in ORT FP32
        ort_sess = make_ort_session(lowered_onnx, threads=2)
        tpl_cases = []
        for s in ref_bundle["samples"]:
            geom = compute_template_geometry(s["tbb"])
            feed = {"z": s["z"], **geom}
            ort_got = dict(zip(ref_bundle["cache_names"], ort_sess.run(None, feed)))
            check = verify_template_caches(ort_got, s["caches"], rel_l2_max=1e-4)
            if not check["passed"]:
                raise RuntimeError(f"Lowered template ORT check failed: {check['max_rel_l2']}")
            tpl_cases.append((feed, s["caches"]))

        _run_pnnx_and_verify(
            lowered_onnx, tpl_dir, pnnx_bin, "template", tpl_cases, None, args.check_fp16
        )
        # Also save raw (z, tbb) inputs so Pi 5 template benchmark times geometry computation inside the loop
        tbb_pack = {f"z_{s['index']}": s["z"] for s in ref_bundle["samples"]}
        tbb_pack.update({f"tbb_{s['index']}": s["tbb"] for s in ref_bundle["samples"]})
        np.savez(tpl_dir / "template_raw_inputs.npz", **tbb_pack)


if __name__ == "__main__":
    main()
