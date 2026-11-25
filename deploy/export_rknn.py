#!/usr/bin/env python3
"""Convert TaCo ONNX graphs to Rockchip RV1126B RKNN FP16 models with numerical verification.

Requires ``rknn-toolkit2`` (tested with v2.3.2) on an x86_64 Linux conversion host.

Supported graphs:
  - ``search``:   converts ``search.onnx`` directly to ``search.rknn`` (FP16, no quantization)
                  and stages NHWC sample inputs + ORT FP32 references for board verification.
  - ``template``: lowers ``template.onnx`` (box geometry -> host CPU, RoIAlign -> MatMul),
                  partitions the FiLM MLP head to ``film_head.npz`` (host CPU FP32), and
                  converts ``template_npu.onnx`` to ``template_npu.rknn`` (FP16).
  - ``both``:     converts both graphs.

Examples:
    python deploy/export_rknn.py --model TaCo-T --out build/rknn/TaCo-T --graph both --verify-sim
    python deploy/export_rknn.py --model TaCo-B --out build/rknn/TaCo-B --graph search
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from template_lowering import FilmHead, compute_template_geometry, lower_template_onnx, split_rknn_template_onnx
from verify_utils import (
    file_sha256,
    generate_ort_references,
    make_ort_session,
    verify_search_outputs,
    verify_template_caches,
)


def _convert_onnx_to_rknn(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str = "rv1126b",
    float_dtype: str = "float16",
    disable_rules=None,
    verify_sim: bool = False,
    sim_inputs_list=None,
):
    from rknn.api import RKNN

    rknn = RKNN(verbose=True, verbose_file=str(rknn_path.with_suffix(".convert.log")))
    cfg_kwargs = {"target_platform": target_platform, "float_dtype": float_dtype}
    if disable_rules:
        cfg_kwargs["disable_rules"] = list(disable_rules)

    t0 = time.time()
    try:
        if rknn.config(**cfg_kwargs) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError(f"rknn.load_onnx failed for {onnx_path}")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError(f"rknn.build failed for {onnx_path}")
        if rknn.export_rknn(str(rknn_path)) != 0:
            raise RuntimeError(f"rknn.export_rknn failed for {rknn_path}")

        sim_outputs = None
        if verify_sim and sim_inputs_list:
            if rknn.init_runtime() != 0:
                raise RuntimeError("rknn.init_runtime (simulator) failed")
            sim_outputs = []
            for feed_list, data_formats in sim_inputs_list:
                sim_outputs.append(rknn.inference(inputs=feed_list, data_format=data_formats))
    finally:
        rknn.release()

    return {
        "rknn_path": str(rknn_path),
        "rknn_bytes": rknn_path.stat().st_size,
        "rknn_sha256": file_sha256(rknn_path),
        "target_platform": target_platform,
        "float_dtype": float_dtype,
        "elapsed_sec": round(time.time() - t0, 2),
    }, sim_outputs


def _to_rknn_input(arr: np.ndarray):
    """4-D tensors are stored NHWC for RKNN runtime; lower-rank tensors keep their shape."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        return np.ascontiguousarray(arr.transpose(0, 2, 3, 1), dtype=np.float32), "nhwc"
    return arr, "nchw"


def export_search_rknn(
    search_onnx: Path,
    out_dir: Path,
    ref_bundle: dict,
    target_platform: str,
    disable_rules,
    verify_sim: bool,
) -> dict:
    search_dir = out_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    cache_names = ref_bundle["cache_names"]
    window = ref_bundle["hann"]

    # Stage sample inputs (NHWC for 4-D) and ORT FP32 references for board/sim verification
    sim_inputs = []
    sample_records = []
    for s in ref_bundle["samples"]:
        sdir = search_dir / "samples" / f"{s['index']:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        tensors = [s["search"]] + [s["caches"][n] for n in cache_names]
        feed_list, fmts, in_files = [], [], []
        for j, t in enumerate(tensors):
            arr, fmt = _to_rknn_input(t)
            fname = f"in_{j:02d}.npy"
            np.save(sdir / fname, arr)
            feed_list.append(arr)
            fmts.append(fmt)
            in_files.append({"name": f"samples/{s['index']:03d}/{fname}", "shape": list(arr.shape), "sha256": file_sha256(sdir / fname)})
        np.savez(
            sdir / "reference.npz",
            score=s["search_refs"]["score"],
            size=s["search_refs"]["size"],
            offset=s["search_refs"]["offset"],
            hann=window,
        )
        sim_inputs.append((feed_list, fmts))
        sample_records.append(
            {
                "sample": f"samples/{s['index']:03d}",
                "inputs": in_files,
                "reference_sha256": file_sha256(sdir / "reference.npz"),
            }
        )

    rknn_path = search_dir / "search.rknn"
    info, sim_outs = _convert_onnx_to_rknn(
        search_onnx,
        rknn_path,
        target_platform=target_platform,
        disable_rules=disable_rules,
        verify_sim=verify_sim,
        sim_inputs_list=sim_inputs,
    )

    sim_report = None
    if sim_outs is not None:
        rows = []
        for s, got in zip(ref_bundle["samples"], sim_outs):
            ref_maps = [s["search_refs"]["score"], s["search_refs"]["size"], s["search_refs"]["offset"]]
            rows.append(verify_search_outputs(got, ref_maps, window))
        sim_report = {
            "passed": all(r["passed"] for r in rows),
            "max_rel_l2": max(r["max_rel_l2"] for r in rows),
            "min_box_iou": min(r["box_iou"] for r in rows),
            "peak_matches": sum(int(r["peak_match"]) for r in rows),
            "num_samples": len(rows),
            "samples": rows,
        }
        print(
            f"[RKNN search sim] passed={sim_report['passed']} "
            f"max_rel_l2={sim_report['max_rel_l2']:.5f} "
            f"min_iou={sim_report['min_box_iou']:.5f} "
            f"peaks={sim_report['peak_matches']}/{sim_report['num_samples']}"
        )

    manifest = {
        "graph": "search",
        "source_onnx_sha256": file_sha256(search_onnx),
        "rknn": "search.rknn",
        **info,
        "input_names": ["search"] + cache_names,
        "output_names": ["score", "size", "offset"],
        "samples": sample_records,
        "simulator_verification": sim_report,
    }
    (search_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_template_rknn(
    template_onnx: Path,
    out_dir: Path,
    ref_bundle: dict,
    target_platform: str,
    disable_rules,
    verify_sim: bool,
) -> dict:
    tpl_dir = out_dir / "template"
    tpl_dir.mkdir(parents=True, exist_ok=True)

    lowered_onnx = tpl_dir / "template_lowered.onnx"
    lower_template_onnx(template_onnx, lowered_onnx, broadcast_pool=False)
    npu_onnx, film_npz, split_meta = split_rknn_template_onnx(lowered_onnx, tpl_dir)

    # Verify lowered + split graph in ORT FP32 before RKNN compilation
    ort_sess = make_ort_session(npu_onnx, threads=2)
    film_head = FilmHead(film_npz)
    npu_in_names = [i.name for i in ort_sess.get_inputs()]

    sim_inputs = []
    sample_records = []
    for s in ref_bundle["samples"]:
        geom = compute_template_geometry(s["tbb"])
        ort_feeds = {"z": s["z"], **geom}
        ort_vals = dict(zip(split_meta["rk_outputs"], ort_sess.run(None, ort_feeds)))
        ort_vals.update(zip(split_meta["film_names"], film_head(ort_vals.pop(split_meta["pooled_name"]))))
        split_check = verify_template_caches(ort_vals, s["caches"], rel_l2_max=1e-4)
        if not split_check["passed"]:
            raise RuntimeError(f"Template lowering/split ORT check failed: max_rel_l2={split_check['max_rel_l2']}")

        sdir = tpl_dir / "samples" / f"{s['index']:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        np.save(sdir / "z.npy", s["z"])
        np.save(sdir / "tbb.npy", s["tbb"])
        np.savez(sdir / "reference.npz", **s["caches"])

        feed_list, fmts = [], []
        for name in npu_in_names:
            arr, fmt = _to_rknn_input(ort_feeds[name])
            feed_list.append(arr)
            fmts.append(fmt)
        sim_inputs.append((feed_list, fmts))
        sample_records.append(
            {
                "sample": f"samples/{s['index']:03d}",
                "z": f"samples/{s['index']:03d}/z.npy",
                "tbb": f"samples/{s['index']:03d}/tbb.npy",
                "reference_sha256": file_sha256(sdir / "reference.npz"),
            }
        )

    rknn_path = tpl_dir / "template_npu.rknn"
    info, sim_outs = _convert_onnx_to_rknn(
        npu_onnx,
        rknn_path,
        target_platform=target_platform,
        disable_rules=disable_rules,
        verify_sim=verify_sim,
        sim_inputs_list=sim_inputs,
    )

    sim_report = None
    if sim_outs is not None:
        rows = []
        for s, got in zip(ref_bundle["samples"], sim_outs):
            out_dict = dict(zip(split_meta["rk_outputs"], got))
            out_dict.update(zip(split_meta["film_names"], film_head(out_dict.pop(split_meta["pooled_name"]))))
            rows.append(verify_template_caches(out_dict, s["caches"]))
        sim_report = {
            "passed": all(r["passed"] for r in rows),
            "max_rel_l2": max(r["max_rel_l2"] for r in rows),
            "num_samples": len(rows),
        }
        print(
            f"[RKNN template sim] passed={sim_report['passed']} "
            f"max_rel_l2={sim_report['max_rel_l2']:.5f}"
        )

    manifest = {
        "graph": "template_hybrid",
        "source_onnx_sha256": file_sha256(template_onnx),
        "rknn": "template_npu.rknn",
        "film_head": "film_head.npz",
        **info,
        **split_meta,
        "samples": sample_records,
        "simulator_verification": sim_report,
    }
    (tpl_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TaCo ONNX models to RKNN FP16 for RV1126B.")
    parser.add_argument("--model", type=str, default="TaCo-T", choices=["TaCo-T", "TaCo-S", "TaCo-B"])
    parser.add_argument("--onnx-dir", type=Path, default=None, help="Directory with template.onnx and search.onnx.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for RKNN artifacts and manifests.")
    parser.add_argument("--graph", type=str, default="both", choices=["search", "template", "both"])
    parser.add_argument("--target", type=str, default="rv1126b", help="Rockchip target platform (default: rv1126b).")
    parser.add_argument("--disable-rules", type=str, default="", help="Optional comma-separated RKNN disable_rules.")
    parser.add_argument("--samples", type=int, default=8, help="Number of verification samples (default: 8).")
    parser.add_argument("--verify-sim", action="store_true", help="Run RKNN simulator numerical check after build.")
    args = parser.parse_args()

    onnx_dir = args.onnx_dir or (REPO_ROOT / "onnx_models" / args.model)
    tpl_onnx = onnx_dir / "template.onnx"
    srch_onnx = onnx_dir / "search.onnx"
    if not tpl_onnx.is_file() or not srch_onnx.is_file():
        raise FileNotFoundError(f"Expected template.onnx and search.onnx inside {onnx_dir}")

    rules = [r.strip() for r in args.disable_rules.split(",") if r.strip()]
    ref_bundle = generate_ort_references(tpl_onnx, srch_onnx, num_samples=args.samples, seed=0)

    if args.graph in ("search", "both"):
        m = export_search_rknn(srch_onnx, args.out, ref_bundle, args.target, rules, args.verify_sim)
        print(f"Built RKNN search graph: {m['rknn_path']} ({m['rknn_bytes'] / 1e6:.2f} MB)")
    if args.graph in ("template", "both"):
        m = export_template_rknn(tpl_onnx, args.out, ref_bundle, args.target, rules, args.verify_sim)
        print(f"Built RKNN hybrid template graph: {m['rknn_path']} ({m['rknn_bytes'] / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
