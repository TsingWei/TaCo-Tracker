#!/usr/bin/env python3
"""Raspberry Pi 5 (Arm Cortex-A76) ncnn benchmark and numerical verification script.

Implements the paper's Raspberry Pi 5 CPU benchmarking protocol:
  - Binds ``--threads`` (default: 4) to CPU cores 0..threads-1.
  - Verifies ncnn FP16 (or FP32) outputs against the ORT FP32 reference bundle
    (``refs.npz``) before timing (rel-L2 <= 0.02, max-abs <= max(0.005, 0.05*peak),
    identical score-peak cell, and decoded-box IoU >= 0.98 for search graphs).
  - Supports ``--screen`` over the paper's 6 ncnn runtime configurations
    (``export_default``, ``winograd_f23``, ``winograd_f43``, ``winograd_off``,
    ``sgemm_off``, ``winograd_sgemm_off``) and selects the fastest passing setting.
  - Supports both ``search`` (per-frame latency) and ``template`` (once-per-video
    latency including host box-geometry weight generation inside the timed call).
  - Samples ``vcgencmd get_throttled`` and CPU frequency during measurement when
    available on Raspberry Pi OS.

Examples:
    # Per-frame search benchmark with 6-setting screen (4 threads, FP16)
    python benchmark/pi5_benchmark.py --model-dir build/ncnn/TaCo-T/search --graph search --threads 4 --screen

    # Template-branch benchmark (4 threads, FP16, geometry included in timing)
    python benchmark/pi5_benchmark.py --model-dir build/ncnn/TaCo-T/template --graph template --threads 4
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from template_lowering import compute_template_geometry
from verify_utils import verify_search_outputs, verify_template_caches

# Six ncnn runtime settings screened in the paper (Sec. C)
SCREEN_SETTINGS: Dict[str, Dict[str, bool]] = {
    "export_default": {},
    "winograd_f23": {"use_winograd_convolution": True, "use_winograd23_convolution": True, "use_winograd43_convolution": False},
    "winograd_f43": {"use_winograd_convolution": True, "use_winograd23_convolution": False, "use_winograd43_convolution": True},
    "winograd_off": {"use_winograd_convolution": False},
    "sgemm_off": {"use_sgemm_convolution": False},
    "winograd_sgemm_off": {"use_winograd_convolution": False, "use_sgemm_convolution": False},
}


def _read_vcgencmd_throttled() -> Optional[str]:
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2).strip()
        return out
    except Exception:
        return None


def _is_currently_throttled(raw: Optional[str]) -> bool:
    if not raw or "=" not in raw:
        return False
    try:
        val = int(raw.split("=", 1)[1], 16)
        return bool(val & 0xF)  # bits 0..3 = active undervoltage / capping / throttling
    except ValueError:
        return False


def _load_ncnn_net(model_dir: Path, fp16: bool, threads: int, overrides: Optional[Dict[str, bool]] = None):
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = int(threads)
    net.opt.use_fp16_packed = bool(fp16)
    net.opt.use_fp16_storage = bool(fp16)
    net.opt.use_fp16_arithmetic = bool(fp16)
    net.opt.use_packing_layout = True
    for key, val in (overrides or {}).items():
        if hasattr(net.opt, key):
            setattr(net.opt, key, val)
    net.load_param(str(model_dir / "model.ncnn.param"))
    net.load_model(str(model_dir / "model.ncnn.bin"))
    return net


def _load_refs(refs_path: Path) -> Tuple[dict, List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]], Optional[np.ndarray]]:
    with np.load(refs_path) as z:
        meta = json.loads(str(z["__meta__"]))
        hann = z["__hann__"].copy() if "__hann__" in z.files else None
        cases = []
        for idx in range(int(meta["num_cases"])):
            feeds = {name: z[f"case{idx}__in__{name}"].copy() for name in meta["input_names"]}
            refs = {name: z[f"case{idx}__out__{name}"].copy() for name in meta["output_names"]}
            cases.append((feeds, refs))
    return meta, cases, hann


def _make_search_call(net, io_map: dict, feeds: Dict[str, np.ndarray]):
    import ncnn

    prepared = [
        (item["ncnn"], np.ascontiguousarray(feeds[item["onnx"]], dtype=np.float32).reshape(item["ncnn_shape"]))
        for item in io_map["inputs"]
    ]
    out_specs = [(item["ncnn"], item["onnx_shape"]) for item in io_map["outputs"]]

    def _call() -> List[np.ndarray]:
        ex = net.create_extractor()
        for blob, arr in prepared:
            ex.input(blob, ncnn.Mat(arr))
        outs = []
        for blob, shape in out_specs:
            ret, mat = ex.extract(blob)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed on {blob} (rc={ret})")
            outs.append(np.array(mat, dtype=np.float32).reshape(shape))
        return outs

    return _call


def _make_template_call(net, io_map: dict, z_crop: np.ndarray, tbb: np.ndarray):
    import ncnn

    in_specs = [(item["onnx"], item["ncnn"], item["ncnn_shape"]) for item in io_map["inputs"]]
    out_specs = [(item["ncnn"], item["onnx_shape"]) for item in io_map["outputs"]]

    def _call() -> List[np.ndarray]:
        # Include box-geometry weight generation inside the timed template call (matches paper Sec. C)
        feeds = {"z": z_crop, **compute_template_geometry(tbb)}
        ex = net.create_extractor()
        for onnx_name, blob, shape in in_specs:
            arr = np.ascontiguousarray(feeds[onnx_name], dtype=np.float32).reshape(shape)
            ex.input(blob, ncnn.Mat(arr))
        outs = []
        for blob, shape in out_specs:
            ret, mat = ex.extract(blob)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed on {blob} (rc={ret})")
            outs.append(np.array(mat, dtype=np.float32).reshape(shape))
        return outs

    return _call


def _benchmark_setting(
    model_dir: Path,
    io_map: dict,
    graph_kind: str,
    cases,
    hann: Optional[np.ndarray],
    raw_templates: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    fp16: bool,
    threads: int,
    setting_name: str,
    overrides: Dict[str, bool],
    rounds: int,
    round_sec: float,
    warmup: int,
) -> dict:
    net = _load_ncnn_net(model_dir, fp16=fp16, threads=threads, overrides=overrides)

    # 1. Numerical verification gate across all reference cases
    gate_rows = []
    for idx, (feed, expected) in enumerate(cases):
        if graph_kind == "search":
            got = _make_search_call(net, io_map, feed)()
            ref_maps = [expected["score"], expected["size"], expected["offset"]]
            gate_rows.append(verify_search_outputs(got, ref_maps, hann))
        else:
            if raw_templates is not None:
                got = _make_template_call(net, io_map, *raw_templates[idx])()
            else:
                got = _make_search_call(net, io_map, feed)()
            got_dict = dict(zip([o["onnx"] for o in io_map["outputs"]], got))
            gate_rows.append(verify_template_caches(got_dict, expected))

    passed = all(r["passed"] for r in gate_rows)
    worst_l2 = max(r["max_rel_l2"] for r in gate_rows)
    if not passed:
        return {
            "setting": setting_name,
            "precision": "fp16" if fp16 else "fp32",
            "passed": False,
            "max_rel_l2": worst_l2,
        }

    # 2. Timed rounds
    if graph_kind == "template" and raw_templates is not None:
        calls = [_make_template_call(net, io_map, z, tbb) for z, tbb in raw_templates]
    else:
        calls = [_make_search_call(net, io_map, cases[0][0])]

    round_means = []
    clean_rounds = 0
    for r in range(rounds):
        th_before = _read_vcgencmd_throttled()
        for i in range(warmup):
            calls[i % len(calls)]()
        samples_ms = []
        deadline = time.perf_counter() + round_sec
        k = 0
        while time.perf_counter() < deadline or len(samples_ms) < 10:
            t0 = time.perf_counter_ns()
            calls[k % len(calls)]()
            samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
            k += 1
        th_after = _read_vcgencmd_throttled()
        is_clean = not (_is_currently_throttled(th_before) or _is_currently_throttled(th_after))
        clean_rounds += int(is_clean)
        round_means.append(statistics.mean(samples_ms))

    median_ms = statistics.median(round_means)
    return {
        "setting": setting_name,
        "precision": "fp16" if fp16 else "fp32",
        "passed": True,
        "max_rel_l2": worst_l2,
        "round_means_ms": [round(v, 4) for v in round_means],
        "median_ms": round(median_ms, 4),
        "fps": round(1000.0 / median_ms, 2),
        "clean_rounds": f"{clean_rounds}/{rounds}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TaCo ncnn models on Raspberry Pi 5 (Arm CPU).")
    parser.add_argument("--model-dir", type=Path, required=True, help="ncnn model directory produced by deploy/export_ncnn.py.")
    parser.add_argument("--graph", type=str, default="search", choices=["search", "template"])
    parser.add_argument("--threads", type=int, default=4, help="Number of ncnn CPU threads (default: 4).")
    parser.add_argument("--fp32", action="store_true", help="Run in FP32 instead of FP16.")
    parser.add_argument("--screen", action="store_true", help="Screen all 6 ncnn runtime settings and report fastest passing.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of timing rounds (default: 5).")
    parser.add_argument("--round-sec", type=float, default=2.6, help="Duration of each timing round in seconds (default: 2.6).")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per round (default: 20).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    # Bind to cores 0..threads-1 on Linux (matches paper's Pi 5 protocol)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(args.threads)))

    io_map = json.loads((args.model_dir / "io_map.json").read_text())
    _, cases, hann = _load_refs(args.model_dir / "refs.npz")

    raw_templates = None
    raw_npz = args.model_dir / "template_raw_inputs.npz"
    if args.graph == "template" and raw_npz.is_file():
        with np.load(raw_npz) as z:
            raw_templates = [(z[f"z_{i}"].copy(), z[f"tbb_{i}"].copy()) for i in range(len(cases))]

    settings = SCREEN_SETTINGS if args.screen else {"export_default": SCREEN_SETTINGS["export_default"]}
    results = []
    for name, ov in settings.items():
        rec = _benchmark_setting(
            args.model_dir,
            io_map,
            args.graph,
            cases,
            hann,
            raw_templates,
            fp16=not args.fp32,
            threads=args.threads,
            setting_name=name,
            overrides=ov,
            rounds=args.rounds,
            round_sec=args.round_sec,
            warmup=args.warmup,
        )
        results.append(rec)
        if rec["passed"]:
            print(
                f"[{name:18s}] PASS | {rec['median_ms']:7.3f} ms ({rec['fps']:6.2f} FPS) | "
                f"max_rel_l2={rec['max_rel_l2']:.4f} | clean={rec['clean_rounds']}"
            )
        else:
            print(f"[{name:18s}] FAIL | max_rel_l2={rec['max_rel_l2']:.4f}")

    passing = [r for r in results if r["passed"]]
    best = min(passing, key=lambda r: r["median_ms"]) if passing else None
    summary = {
        "model_dir": str(args.model_dir),
        "graph": args.graph,
        "threads": args.threads,
        "precision": "fp32" if args.fp32 else "fp16",
        "selected": best,
        "all_settings": results,
    }
    if best is not None:
        print(f"\nSelected setting: {best['setting']} -> {best['median_ms']:.3f} ms ({best['fps']:.2f} FPS)")
    else:
        print("\nNo setting passed the numerical verification gate.", file=sys.stderr)
        sys.exit(1)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
