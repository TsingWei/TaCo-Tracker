#!/usr/bin/env python3
"""Minimal benchmark evaluation script for TaCo (supports PyTorch and ONNX Runtime).

Preprocessing and data loading use pure NumPy and OpenCV for both backends.
When ``--backend onnx`` is selected, PyTorch is never imported.

Usage:
    # PyTorch backend (uses bundled weights/<model>.pth; CPU or CUDA GPU)
    python eval.py TaCo-T --dataset lasot --backend pytorch --provider cuda --fp16 --threads 4 --num_gpus 1
    python eval.py TaCo-S --dataset lasot --backend pytorch --provider cpu --sequence airplane-1 --debug 1

    # ONNX Runtime backend (zero PyTorch; export ONNX first via deploy/export_onnx.py)
    python eval.py TaCo-T --dataset lasot --backend onnx --threads 4 --num_gpus 1
"""

import argparse
import hashlib
import io
import multiprocessing
import os
import sys
import time
import zipfile
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from config import DATASET_PATHS, MODELS, RESULTS_PATH
from datasets import get_dataset
from tracker import create_tracker

NESTED_DATASETS = {
    "trackingnet",
    "got10k",
    "lasot",
    "lasot_extension_subset",
    "uav",
    "nfs",
    "tnl2k",
}


def _read_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Failed to read frame image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _result_base_path(seq, results_dir: str) -> str:
    if seq.dataset in NESTED_DATASETS:
        dataset_dir = os.path.join(results_dir, seq.dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        return os.path.join(dataset_dir, seq.name)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, seq.name)


def _save_output(seq, results_dir: str, output: dict) -> None:
    base = _result_base_path(seq, results_dir)
    if output.get("target_bbox"):
        np.savetxt(f"{base}.txt", np.asarray(output["target_bbox"], dtype=np.float64), delimiter="\t", fmt="%.2f")
    if output.get("time"):
        np.savetxt(f"{base}_time.txt", np.asarray(output["time"], dtype=np.float64), delimiter="\t", fmt="%.6f")


def _run_sequence(
    seq,
    model_cfg: dict,
    results_dir: str,
    debug: bool,
    num_gpus: int,
    backend: str,
    provider: str,
    fp16: bool,
    runtime_threads: Optional[int],
    weights: Optional[str] = None,
) -> Optional[float]:
    gpu_id = 0
    if provider != "cpu" and num_gpus > 0:
        worker_name = multiprocessing.current_process().name
        if worker_name != "MainProcess":
            try:
                suffix = worker_name.rsplit("-", 1)[-1]
                gpu_id = (int(suffix) - 1) % int(num_gpus)
            except Exception:
                gpu_id = 0

    if not debug:
        bbox_file = f"{_result_base_path(seq, results_dir)}.txt"
        if os.path.isfile(bbox_file):
            return None

    tracker = create_tracker(
        model_cfg,
        backend=backend,
        device_id=gpu_id,
        provider=provider,
        fp16=fp16,
        num_threads=runtime_threads,
        weights=weights,
    )
    output = {"target_bbox": [], "time": []}

    image = _read_rgb(seq.frames[0])
    init_info = seq.init_info()
    t0 = time.perf_counter()
    tracker.initialize(image, init_info)
    output["target_bbox"].append(init_info["init_bbox"])
    output["time"].append(time.perf_counter() - t0)

    for frame_path in seq.frames[1:]:
        image = _read_rgb(frame_path)
        t0 = time.perf_counter()
        out = tracker.track(image)
        elapsed = time.perf_counter() - t0
        output["target_bbox"].append(out["target_bbox"])
        output["time"].append(elapsed)

    total_time = sum(output["time"])
    fps = len(output["time"]) / total_time if total_time > 0 else 0.0

    if not debug:
        _save_output(seq, results_dir, output)
    return fps


def _worker_fn(args):
    seq, model_cfg, results_dir, debug, num_gpus, backend, provider, fp16, runtime_threads, weights = args
    try:
        fps = _run_sequence(seq, model_cfg, results_dir, debug, num_gpus, backend, provider, fp16, runtime_threads, weights)
        return seq.name, fps, None
    except Exception as exc:
        return seq.name, None, repr(exc)


def _resolve_dataset_paths(dataset_name: str, data_root: Optional[str]) -> dict:
    paths = dict(DATASET_PATHS)
    if not data_root:
        return paths
    key_map = {
        "lasot": "lasot_path",
        "lasot_extension_subset": "lasot_extension_subset_path",
        "got10k_test": "got10k_path",
        "got10k_val": "got10k_path",
        "trackingnet": "trackingnet_path",
        "vasttrack_test": "vasttrack_test_path",
        "uav": "uav_path",
        "uavdt": "uavdt_path",
        "dtb70": "dtb70_path",
        "nfs": "nfs_path",
        "nfs_30": "nfs_path",
        "nfs_60": "nfs_path",
        "nfs_120": "nfs_path",
        "nfs_240": "nfs_path",
        "tnl2k": "tnl2k_path",
        "visdrone2018": "visdrone2018_path",
        "uavdark135": "uavdark135_path",
    }
    key = key_map.get(dataset_name.lower())
    if key:
        paths[key] = data_root
    return paths


def _pack_submission_zip(dataset_name: str, dataset, results_dir: str, model_name: str) -> str:
    """Package predictions for official GOT-10k or TrackingNet server submission."""
    ds_key = dataset_name.lower()
    zip_path = os.path.join(results_dir, f"{model_name}_{ds_key}_submission.zip")
    tmp_path = f"{zip_path}.tmp"
    os.makedirs(results_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for seq in dataset:
                base = _result_base_path(seq, results_dir)
                bbox_path = f"{base}.txt"
                if not os.path.isfile(bbox_path):
                    raise FileNotFoundError(f"Missing prediction file for packaging: {bbox_path}")
                boxes = np.loadtxt(bbox_path, dtype=np.float64, ndmin=2)
                expected_frames = len(seq.frames)
                if boxes.shape != (expected_frames, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
                    raise ValueError(f"Invalid prediction boxes in {bbox_path}: shape={boxes.shape}, expected=({expected_frames}, 4)")
                buf = io.StringIO()
                np.savetxt(buf, boxes, delimiter=",", fmt="%.3f")
                if ds_key in ("got10k_test", "got10k_val"):
                    time_path = f"{base}_time.txt"
                    if not os.path.isfile(time_path):
                        raise FileNotFoundError(f"Missing timing file for GOT-10k packaging: {time_path}")
                    times = np.loadtxt(time_path, dtype=np.float64, ndmin=1)
                    if times.shape != (expected_frames,) or not np.isfinite(times).all():
                        raise ValueError(f"Invalid timing rows in {time_path}: shape={times.shape}, expected=({expected_frames},)")
                    tbuf = io.StringIO()
                    np.savetxt(tbuf, times, fmt="%.6f")
                    zf.writestr(f"{seq.name}/{seq.name}_001.txt", buf.getvalue())
                    zf.writestr(f"{seq.name}/{seq.name}_time.txt", tbuf.getvalue())
                else:
                    zf.writestr(f"{seq.name}.txt", buf.getvalue())
        with zipfile.ZipFile(tmp_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt submission archive member: {bad_member}")
        os.replace(tmp_path, zip_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TaCo trackers on benchmark datasets (NumPy preprocessing).")
    parser.add_argument("model_name", type=str, choices=list(MODELS.keys()), help="TaCo model variant.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., lasot, got10k_test, trackingnet, vasttrack_test, uav).")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx"], help="Inference backend (default: auto -> onnx if exported, else pytorch).")
    parser.add_argument("--provider", "--device", dest="provider", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Execution device/provider (auto, cuda, or cpu).")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 autocast (PyTorch CUDA backend only; matches paper evaluation).")
    parser.add_argument("--weights", help="Evaluate a TaCo .pth checkpoint with the PyTorch backend.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path from config.py.")
    parser.add_argument("--results-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--threads", type=int, default=0, help="Number of worker processes (0 = sequential).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for round-robin worker assignment.")
    parser.add_argument("--ort-threads", "--cpu-threads", dest="runtime_threads", type=int, default=None, help="CPU threads per worker (ORT intra-op or PyTorch CPU threads).")
    parser.add_argument("--sequence", type=str, default=None, help="Run only one sequence by name.")
    parser.add_argument("--debug", type=int, default=0, help="1 = do not save prediction files.")
    parser.add_argument("--pack-zip", action="store_true", help="Pack predictions into an official server submission ZIP (got10k_test, got10k_val, or trackingnet).")
    args = parser.parse_args()
    if args.weights and args.backend == "onnx":
        parser.error("--weights requires --backend pytorch or auto")
    if args.pack_zip:
        if args.dataset.lower() not in ("got10k_test", "got10k_val", "trackingnet"):
            parser.error("--pack-zip only supports got10k_test, got10k_val, or trackingnet")
        if args.sequence is not None:
            parser.error("--pack-zip cannot be used with --sequence")
        if args.debug:
            parser.error("--pack-zip cannot be used with --debug 1")

    model_cfg = MODELS[args.model_name]
    if args.weights:
        args.backend = "pytorch"
        args.weights = os.path.expanduser(args.weights)
        if not os.path.isfile(args.weights):
            parser.error(f"PyTorch weights not found: {args.weights}")
    results_model = args.model_name
    if args.weights and args.results_dir is None and os.path.abspath(args.weights) != os.path.abspath(model_cfg["weights"]):
        # A custom checkpoint can reuse a filename such as checkpoint_latest.pth.
        digest = hashlib.sha256()
        with open(args.weights, "rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        stem = os.path.splitext(os.path.basename(args.weights))[0]
        results_model = os.path.join(results_model, f"{stem}-{digest.hexdigest()[:12]}")
    results_dir = args.results_dir or os.path.join(RESULTS_PATH, results_model, args.dataset)
    dataset_paths = _resolve_dataset_paths(args.dataset, args.data_root)
    dataset = get_dataset(args.dataset, dataset_paths)

    if args.sequence is not None:
        matched = [s for s in dataset if s.name == args.sequence]
        if not matched:
            print(f"Sequence {args.sequence!r} not found in dataset {args.dataset!r}.", file=sys.stderr)
            sys.exit(1)
        dataset = matched

    print(f"Model    : {args.model_name}")
    print(f"Dataset  : {args.dataset} ({len(dataset)} sequence(s))")
    print(f"Backend  : {args.backend} | device={args.provider} | fp16={args.fp16} (num_gpus={args.num_gpus}, workers={args.threads})")
    print(f"Results  : {results_dir}")

    fps_list = []
    if args.threads <= 0 or len(dataset) == 1:
        iterator = tqdm(dataset, desc="Evaluating", unit="seq") if (tqdm and len(dataset) > 1) else dataset
        for seq in iterator:
            fps = _run_sequence(
                seq, model_cfg, results_dir, bool(args.debug), args.num_gpus,
                args.backend, args.provider, args.fp16, args.runtime_threads,
                args.weights,
            )
            if fps is not None:
                fps_list.append(fps)
                if tqdm is None or len(dataset) == 1:
                    print(f"  {seq.name}: {fps:.1f} FPS")
    else:
        multiprocessing.set_start_method("spawn", force=True)
        tasks = [
            (seq, model_cfg, results_dir, bool(args.debug), args.num_gpus,
             args.backend, args.provider, args.fp16, args.runtime_threads, args.weights)
            for seq in dataset
        ]
        with multiprocessing.Pool(processes=args.threads) as pool:
            iterator = pool.imap_unordered(_worker_fn, tasks)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(tasks), desc="Evaluating", unit="seq")
            for name, fps, err in iterator:
                if err is not None:
                    print(f"Error on {name}: {err}", file=sys.stderr)
                elif fps is not None:
                    fps_list.append(fps)
                    if tqdm is None:
                        print(f"  {name}: {fps:.1f} FPS")

    if fps_list:
        print(f"Done. Mean sequence FPS: {sum(fps_list) / len(fps_list):.2f}")
    else:
        print("Done (all sequences already existed or skipped).")

    if args.pack_zip:
        zip_path = _pack_submission_zip(args.dataset, dataset, results_dir, args.model_name)
        print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
