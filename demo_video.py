#!/usr/bin/env python3
"""Minimal video and image-sequence inference demo for TaCo (PyTorch or ONNX Runtime).

Preprocessing and video/frame reading use pure NumPy and OpenCV for both backends.
When ``--backend onnx`` is selected, PyTorch is never imported.

Examples:
    # PyTorch GPU or CPU using bundled weights/TaCo-T.pth
    python demo_video.py TaCo-T --backend pytorch --device cuda --input sample.mp4 --init-bbox 120,80,64,96 --output tracked.mp4
    python demo_video.py TaCo-S --backend pytorch --device cpu --input /path/to/seq/img --init-bbox 120,80,64,96 --save-bbox pred.txt

    # ONNX Runtime (zero PyTorch; export ONNX first via deploy/export_onnx.py)
    python demo_video.py TaCo-T --backend onnx --input sample.mp4 --init-bbox 120,80,64,96 --output tracked.mp4
"""

import argparse
import os
import sys
import time
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from config import MODELS
from tracker import create_tracker

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _parse_bbox(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"--init-bbox must have 4 values (x,y,w,h), got: {raw!r}")
    bbox = [float(v) for v in parts]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Bounding box width and height must be positive, got: {bbox}")
    return bbox


def _iter_frames(source: str) -> Tuple[Iterator[np.ndarray], float, Optional[int]]:
    """Yield BGR frames from a video file or sorted image directory."""
    if os.path.isdir(source):
        files = sorted(
            os.path.join(source, f)
            for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No image frames found in directory: {source}")

        def _gen_dir():
            for path in files:
                frame = cv2.imread(path)
                if frame is None:
                    raise RuntimeError(f"Failed to decode image: {path}")
                yield frame

        return _gen_dir(), 30.0, len(files)

    if not os.path.isfile(source):
        raise FileNotFoundError(f"Input video or directory not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None

    def _gen_video():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return _gen_video(), fps, total


def _draw_overlay(frame_bgr: np.ndarray, bbox: List[float], score: float, fps: float, frame_idx: int) -> np.ndarray:
    vis = frame_bgr.copy()
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
    label = f"#{frame_idx}  score={score:.2f}  {fps:.1f} FPS"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TaCo tracker on a video file or frame folder (NumPy preprocessing).")
    parser.add_argument("model_name", type=str, choices=list(MODELS.keys()), help="TaCo model variant.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file or directory of frames.")
    parser.add_argument("--init-bbox", type=str, default=None, help="Initial bounding box 'x,y,w,h' on frame 0.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx"], help="Inference backend (default: auto -> onnx if exported, else pytorch).")
    parser.add_argument("--provider", "--device", dest="provider", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Execution device/provider (auto, cuda, or cpu).")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 autocast (PyTorch CUDA backend only).")
    parser.add_argument("--weights", help="Run a TaCo .pth checkpoint with the PyTorch backend.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated output video (.mp4).")
    parser.add_argument("--save-bbox", type=str, default=None, help="Optional path to save per-frame boxes (x,y,w,h).")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id when using GPU.")
    parser.add_argument("--ort-threads", "--cpu-threads", dest="runtime_threads", type=int, default=None, help="CPU threads (ORT intra-op or PyTorch CPU threads).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames to process.")
    parser.add_argument("--show", action="store_true", help="Display live tracking window (press 'q' or ESC to quit).")
    args = parser.parse_args()
    if args.weights and args.backend == "onnx":
        parser.error("--weights requires --backend pytorch or auto")

    frames_iter, video_fps, total_frames = _iter_frames(args.input)
    try:
        first_bgr = next(frames_iter)
    except StopIteration:
        print("Input contains zero frames.", file=sys.stderr)
        sys.exit(1)

    if args.init_bbox is not None:
        init_bbox = _parse_bbox(args.init_bbox)
    else:
        roi = cv2.selectROI("Select Target (TaCo)", first_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target (TaCo)")
        init_bbox = [float(v) for v in roi]
        if init_bbox[2] <= 0 or init_bbox[3] <= 0:
            print("No target selected; exiting.", file=sys.stderr)
            sys.exit(1)

    tracker = create_tracker(
        MODELS[args.model_name],
        backend=args.backend,
        device_id=args.device_id,
        provider=args.provider,
        fp16=args.fp16,
        num_threads=args.runtime_threads,
        weights=args.weights,
    )

    writer = None
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        h, w = first_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, video_fps, (w, h))

    # Initialize on frame 0
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    tracker.initialize(first_rgb, {"init_bbox": init_bbox})
    init_ms = (time.perf_counter() - t0) * 1000.0

    boxes: List[List[float]] = [init_bbox]
    track_times: List[float] = []

    if writer is not None or args.show:
        vis = _draw_overlay(first_bgr, init_bbox, 1.0, 1000.0 / max(init_ms, 1e-3), 0)
        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow(f"TaCo ({args.model_name})", vis)

    frame_idx = 0
    try:
        for frame_bgr in frames_iter:
            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            out = tracker.track(frame_rgb)
            dt = time.perf_counter() - t0
            track_times.append(dt)
            boxes.append(out["target_bbox"])

            if writer is not None or args.show:
                vis = _draw_overlay(
                    frame_bgr, out["target_bbox"], out["best_score"], 1.0 / max(dt, 1e-6), frame_idx
                )
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow(f"TaCo ({args.model_name})", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.save_bbox:
        out_dir = os.path.dirname(os.path.abspath(args.save_bbox))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savetxt(args.save_bbox, np.asarray(boxes, dtype=np.float64), delimiter=",", fmt="%.2f")

    mean_ms = (sum(track_times) / len(track_times) * 1000.0) if track_times else 0.0
    mean_fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
    total_str = f"/{total_frames}" if total_frames is not None else ""
    print(
        f"Tracked {len(boxes)}{total_str} frames | init: {init_ms:.2f} ms | "
        f"per-frame: {mean_ms:.2f} ms ({mean_fps:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
