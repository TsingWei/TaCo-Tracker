#!/usr/bin/env python3
"""Rockchip RV1126B NPU benchmark and numerical verification script for TaCo.

Supports both per-frame search benchmarking and hybrid (NPU + CPU FP32) template
benchmarking, implementing the paper's exact RV1126B measurement protocol:
  - Per-frame search graph (``--graph search``):
      1. Numerical gate over all staged samples against ORT FP32
         (rel-L2 <= 0.02, identical score-peak cell, decoded-box IoU >= 0.98).
      2. Warm per-operator profile via ``RKNN_QUERY_PERF_DETAIL`` checking that
         zero compute operators fall back to the CPU.
      3. Timing over 5 rounds of 1,000 ``rknn_run`` calls (after 5 warmups),
         reporting the median of round means. Uses Rockchip's C ``rknn_benchmark``
         binary when ``--cbench`` is provided (exact paper protocol), or falls back
         to direct ``librknnrt.so`` ctypes timing when omitted.
  - Template branch (``--graph template``):
      1. Runs the hybrid pipeline (host box geometry + RKNN NPU backbone & kernel
         generation + host CPU FP32 FiLM head) across all staged samples and verifies
         every cache tensor against ORT FP32 (rel-L2 <= 0.02).
      2. Times 5 rounds of 2 s each (after 20 warmups), reporting the median of
         round means.

Examples:
    # Per-frame NPU timing using Rockchip's C rknn_benchmark binary
    python benchmark/rv1126b_benchmark.py --model-dir build/rknn/TaCo-T/search \\
        --graph search --cbench /path/to/rknn_benchmark_aarch64 --disable-cpuidle

    # Hybrid template-branch timing on RV1126B
    python benchmark/rv1126b_benchmark.py --model-dir build/rknn/TaCo-T/template \\
        --graph template --disable-cpuidle
"""

import argparse
import contextlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from rknn_runtime import RknnHybridTemplateRunner, RknnRunner, parse_rknn_perf
from verify_utils import verify_search_outputs, verify_template_caches

NPU_DEVFREQ = Path("/sys/class/devfreq/22000000.npu")
CPU_ROOT = Path("/sys/devices/system/cpu")


def _read_sysfs(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def read_telemetry() -> dict:
    npu_freq = {k: _read_sysfs(NPU_DEVFREQ / k) for k in ("cur_freq", "min_freq", "max_freq")}
    temps = {p.parent.name: _read_sysfs(p) for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))}
    govs = {p.parent.parent.name: _read_sysfs(p) for p in sorted(CPU_ROOT.glob("cpu[0-9]/cpufreq/scaling_governor"))}
    return {"npu_freq_hz": npu_freq, "cpu_governors": govs, "thermal_millidegc": temps}


@contextlib.contextmanager
def _maybe_disable_cpuidle(enabled: bool):
    if not enabled:
        yield
        return
    idle_files = sorted(CPU_ROOT.glob("cpu[0-9]/cpuidle/state1/disable"))
    if not idle_files:
        print("Warning: cpuidle state1 sysfs nodes not found; proceeding without cpuidle change.", file=sys.stderr)
        yield
        return
    original = {p: p.read_text() for p in idle_files}
    try:
        for p in idle_files:
            p.write_text("1\n")
        yield
    finally:
        for p, val in original.items():
            p.write_text(val)


def _parse_cbench_log(text: str, expected_iters: int) -> dict:
    perf_section = text.split("Begin perf", 1)[1] if "Begin perf" in text else text
    timings = [float(v) for v in re.findall(r"^\s*\d+:\s*Elapse Time\s*=\s*([\d.]+)ms", perf_section, re.M)]
    fps_match = re.search(r"Avg FPS\s*=\s*([\d.]+)", text)
    if not fps_match:
        raise RuntimeError("Could not parse 'Avg FPS' from rknn_benchmark output.")
    fps = float(fps_match.group(1))
    return {
        "calls": len(timings) if timings else expected_iters,
        "mean_ms": 1000.0 / fps,
        "fps": fps,
        "median_ms": statistics.median(timings) if timings else (1000.0 / fps),
    }


def benchmark_search(
    model_dir: Path,
    lib_path: str,
    cbench_bin: Optional[Path],
    rounds: int,
    iters: int,
    warmup: int,
    profile_warm: int,
) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    rknn_path = model_dir / manifest["rknn"]

    # 1. Numerical verification across all staged samples
    gate_rows = []
    if cbench_bin is not None:
        for item in manifest["samples"]:
            sdir = model_dir / item["sample"]
            inputs = [str(model_dir / spec["name"]) for spec in item["inputs"]]
            res = subprocess.run(
                [str(cbench_bin), str(rknn_path), "#".join(inputs), "1", "0"],
                cwd=str(sdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            if "Avg FPS" not in res.stdout:
                raise RuntimeError(f"cbench verification failed on {sdir}:\n{res.stdout}")
            with np.load(sdir / "reference.npz") as ref:
                ref_maps = [ref["score"], ref["size"], ref["offset"]]
                act_maps = [np.load(sdir / f"rt_output{i}.npy") for i in range(3)]
                gate_rows.append(verify_search_outputs(act_maps, ref_maps, ref["hann"]))
    else:
        runner = RknnRunner(rknn_path, lib_path=lib_path, collect_perf=False)
        try:
            for item in manifest["samples"]:
                sdir = model_dir / item["sample"]
                in_arrays = [np.load(model_dir / spec["name"]) for spec in item["inputs"]]
                act_maps = runner.run(in_arrays)
                with np.load(sdir / "reference.npz") as ref:
                    ref_maps = [ref["score"], ref["size"], ref["offset"]]
                    gate_rows.append(verify_search_outputs(act_maps, ref_maps, ref["hann"]))
        finally:
            runner.close()

    # 2. Warm per-op profile to verify 0 CPU fallback operators
    first_inputs = [np.load(model_dir / spec["name"]) for spec in manifest["samples"][0]["inputs"]]
    prof_runner = RknnRunner(rknn_path, lib_path=lib_path, collect_perf=True)
    try:
        prof_runner.set_inputs(first_inputs)
        for _ in range(profile_warm):
            prof_runner.run_only()
        prof_runner.get_outputs()
        perf_text = prof_runner.query_perf_detail()
    finally:
        prof_runner.close()
    ops = parse_rknn_perf(perf_text)
    cpu_ops = [op for op in ops if op["target"] == "CPU" and op["op"] not in {"InputOperator", "OutputOperator"}]

    gate_passed = (len(cpu_ops) == 0) and all(r["passed"] for r in gate_rows)
    gate_summary = {
        "passed": gate_passed,
        "max_rel_l2": max(r["max_rel_l2"] for r in gate_rows),
        "min_box_iou": min(r["box_iou"] for r in gate_rows),
        "peak_matches": sum(int(r["peak_match"]) for r in gate_rows),
        "num_samples": len(gate_rows),
        "operator_count": len(ops),
        "cpu_fallback_ops": len(cpu_ops),
    }
    print(
        f"[RV1126B search gate] passed={gate_passed} | max_rel_l2={gate_summary['max_rel_l2']:.5f} | "
        f"min_iou={gate_summary['min_box_iou']:.5f} | peaks={gate_summary['peak_matches']}/{gate_summary['num_samples']} | "
        f"cpu_fallback={len(cpu_ops)}/{len(ops)}"
    )
    if not gate_passed:
        raise RuntimeError(f"RV1126B numerical or CPU-fallback gate failed: {gate_summary}")

    # 3. Timing rounds
    round_rows = []
    if cbench_bin is not None:
        in_arg = "#".join(str(model_dir / spec["name"]) for spec in manifest["samples"][0]["inputs"])
        for r in range(1, rounds + 1):
            proc = subprocess.run(
                [str(cbench_bin), str(rknn_path), in_arg, str(iters), "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            row = {"round": r, **_parse_cbench_log(proc.stdout, iters)}
            round_rows.append(row)
            print(f"  round {r}/{rounds}: mean={row['mean_ms']:.3f} ms ({row['fps']:.2f} FPS)")
    else:
        runner = RknnRunner(rknn_path, lib_path=lib_path, collect_perf=False)
        try:
            runner.set_inputs(first_inputs)
            for r in range(1, rounds + 1):
                for _ in range(warmup):
                    runner.run_only()
                t0 = time.perf_counter_ns()
                for _ in range(iters):
                    runner.run_only()
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                mean_ms = elapsed_ms / iters
                row = {"round": r, "calls": iters, "mean_ms": mean_ms, "fps": 1000.0 / mean_ms}
                round_rows.append(row)
                print(f"  round {r}/{rounds}: mean={mean_ms:.3f} ms ({row['fps']:.2f} FPS)")
            runner.get_outputs()
        finally:
            runner.close()

    median_ms = statistics.median(r["mean_ms"] for r in round_rows)
    return {
        "graph": "search",
        "timing_backend": "cbench" if cbench_bin is not None else "ctypes_librknnrt",
        "gate": gate_summary,
        "median_ms": round(median_ms, 4),
        "fps": round(1000.0 / median_ms, 2),
        "rounds": round_rows,
        "telemetry": read_telemetry(),
    }


def benchmark_template(
    model_dir: Path,
    lib_path: str,
    rounds: int,
    round_sec: float,
    warmup: int,
) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    runner = RknnHybridTemplateRunner(model_dir, lib_path=lib_path)
    try:
        inputs = []
        gate_rows = []
        for item in manifest["samples"]:
            z = np.load(model_dir / item["z"])
            tbb = np.load(model_dir / item["tbb"])
            inputs.append((z, tbb))
            caches = runner.run(z, tbb)
            with np.load(model_dir / item["sample"] / "reference.npz") as ref:
                ref_dict = {k: ref[k] for k in ref.files}
            gate_rows.append(verify_template_caches(caches, ref_dict))

        passed = all(r["passed"] for r in gate_rows)
        worst_l2 = max(r["max_rel_l2"] for r in gate_rows)
        print(f"[RV1126B template gate] passed={passed} | max_rel_l2={worst_l2:.5f} ({len(gate_rows)} samples)")
        if not passed:
            raise RuntimeError(f"RV1126B hybrid template cache check failed: max_rel_l2={worst_l2}")

        round_rows = []
        for r in range(1, rounds + 1):
            for i in range(warmup):
                runner.run(*inputs[i % len(inputs)])
            elapsed = []
            deadline = time.perf_counter() + round_sec
            while time.perf_counter() < deadline or len(elapsed) < 10:
                t0 = time.perf_counter_ns()
                runner.run(*inputs[len(elapsed) % len(inputs)])
                elapsed.append((time.perf_counter_ns() - t0) / 1e6)
            mean_ms = statistics.mean(elapsed)
            row = {
                "round": r,
                "calls": len(elapsed),
                "mean_ms": mean_ms,
                "median_ms": statistics.median(elapsed),
                "p95_ms": sorted(elapsed)[int(len(elapsed) * 0.95)],
            }
            round_rows.append(row)
            print(f"  round {r}/{rounds}: mean={mean_ms:.3f} ms ({len(elapsed)} calls)")
    finally:
        runner.close()

    median_ms = statistics.median(r["mean_ms"] for r in round_rows)
    return {
        "graph": "template_hybrid",
        "gate": {"passed": passed, "max_rel_l2": worst_l2, "num_samples": len(gate_rows)},
        "median_ms": round(median_ms, 4),
        "rounds": round_rows,
        "telemetry": read_telemetry(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and verify TaCo RKNN models on Rockchip RV1126B NPU.")
    parser.add_argument("--model-dir", type=Path, required=True, help="RKNN artifact dir produced by deploy/export_rknn.py.")
    parser.add_argument("--graph", type=str, default="search", choices=["search", "template"])
    parser.add_argument("--cbench", type=Path, default=None, help="Path to Rockchip C rknn_benchmark binary (recommended for search).")
    parser.add_argument("--rknn-lib", type=str, default="/usr/lib/librknnrt.so", help="Path to librknnrt.so on the board.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of timing rounds (default: 5).")
    parser.add_argument("--iters", type=int, default=1000, help="Calls per round for search timing (default: 1000).")
    parser.add_argument("--round-sec", type=float, default=2.0, help="Seconds per round for template timing (default: 2.0).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per round (default: 5 for search, 20 recommended for template).")
    parser.add_argument("--profile-warm", type=int, default=100, help="Warmup runs before PERF_DETAIL operator query.")
    parser.add_argument("--disable-cpuidle", action="store_true", help="Temporarily disable CPU state1 deep idle during measurement.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    with _maybe_disable_cpuidle(args.disable_cpuidle):
        if args.graph == "search":
            summary = benchmark_search(
                args.model_dir, args.rknn_lib, args.cbench, args.rounds, args.iters, args.warmup, args.profile_warm
            )
            print(f"\nRV1126B NPU search latency: {summary['median_ms']:.3f} ms ({summary['fps']:.2f} FPS)")
        else:
            warmup = args.warmup if args.warmup != 5 else 20
            summary = benchmark_template(args.model_dir, args.rknn_lib, args.rounds, args.round_sec, warmup)
            print(f"\nRV1126B hybrid template latency: {summary['median_ms']:.3f} ms (once per video)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
