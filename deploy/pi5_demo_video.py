#!/usr/bin/env python3
"""Raspberry Pi 5 (Arm Cortex-A76) video / image-folder / camera inference with ncnn.

Runs the two-branch ncnn graphs exported by ``deploy/export_ncnn.py`` directly on a
Raspberry Pi 5 (or any host with Python ``ncnn``, ``numpy``, and ``opencv-python``)
with zero PyTorch or ONNX Runtime dependency.

Step 1 (on your conversion host):
    python deploy/export_onnx.py --model TaCo-T --weights weights/TaCo-T.pth
    python deploy/export_ncnn.py --model TaCo-T --out build/ncnn/TaCo-T --graph both --check-fp16

Step 2 (copy build/ncnn/TaCo-T/, deploy/, and tracker.py to Pi 5):
    pip install ncnn numpy opencv-python
    python deploy/pi5_demo_video.py \
        --model-dir build/ncnn/TaCo-T \
        --input sample.mp4 \
        --init-bbox 120,80,96,64 \
        --output tracked_pi5.mp4 \
        --threads 4

    # Optional: pass --setting winograd_f23 (fastest passing setting for TaCo-T on Pi 5)
    # Optional: pass --input 0 --show for live USB/CSI camera tracking
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from template_lowering import compute_template_geometry
from tracker import _BaseTaCoTracker

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Same 6 ncnn runtime configurations screened in benchmark/pi5_benchmark.py
NCNN_SETTINGS: Dict[str, Dict[str, bool]] = {
    "export_default": {},
    "winograd_f23": {
        "use_winograd_convolution": True,
        "use_winograd23_convolution": True,
        "use_winograd43_convolution": False,
    },
    "winograd_f43": {
        "use_winograd_convolution": True,
        "use_winograd23_convolution": False,
        "use_winograd43_convolution": True,
    },
    "winograd_off": {"use_winograd_convolution": False},
    "sgemm_off": {"use_sgemm_convolution": False},
    "winograd_sgemm_off": {
        "use_winograd_convolution": False,
        "use_sgemm_convolution": False,
    },
}


def _load_ncnn_net(model_dir: Path, fp16: bool, threads: int, setting: str):
    import ncnn

    param_path = model_dir / "model.ncnn.param"
    bin_path = model_dir / "model.ncnn.bin"
    if not param_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(
            f"Missing model.ncnn.param or model.ncnn.bin in {model_dir}. "
            "Export with: python deploy/export_ncnn.py --model <variant> --out <dir> --graph both"
        )

    net = ncnn.Net()
    net.opt.num_threads = int(threads)
    net.opt.use_fp16_packed = bool(fp16)
    net.opt.use_fp16_storage = bool(fp16)
    net.opt.use_fp16_arithmetic = bool(fp16)
    net.opt.use_packing_layout = True
    for key, val in NCNN_SETTINGS[setting].items():
        if hasattr(net.opt, key):
            setattr(net.opt, key, val)

    if net.load_param(str(param_path)) != 0:
        raise RuntimeError(f"Failed to load ncnn param: {param_path}")
    if net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"Failed to load ncnn model: {bin_path}")
    return net


class TaCoNCNNTracker(_BaseTaCoTracker):
    """PyTorch-free TaCo tracker backed by ncnn for Raspberry Pi 5 / Arm CPUs."""

    def __init__(
        self,
        model_dir: Path,
        threads: int = 4,
        fp16: bool = True,
        setting: str = "export_default",
    ):
        super().__init__(
            {
                "template_size": 128,
                "search_size": 256,
                "template_factor": 2.0,
                "search_factor": 4.0,
                "stride": 16,
            }
        )
        import ncnn

        self._ncnn = ncnn
        tpl_dir = model_dir / "template"
        srch_dir = model_dir / "search"

        tpl_io_path = tpl_dir / "io_map.json"
        srch_io_path = srch_dir / "io_map.json"
        if not tpl_io_path.is_file() or not srch_io_path.is_file():
            raise FileNotFoundError(
                f"Expected template/io_map.json and search/io_map.json under {model_dir}. "
                "Export both graphs with: python deploy/export_ncnn.py --model <variant> --out <dir> --graph both"
            )

        self.tpl_io = json.loads(tpl_io_path.read_text())
        self.srch_io = json.loads(srch_io_path.read_text())

        self.tpl_net = _load_ncnn_net(tpl_dir, fp16=fp16, threads=threads, setting="export_default")
        self.srch_net = _load_ncnn_net(srch_dir, fp16=fp16, threads=threads, setting=setting)

        self._tpl_inputs = [
            (item["onnx"], item["ncnn"], tuple(item["ncnn_shape"])) for item in self.tpl_io["inputs"]
        ]
        self._tpl_outputs = [
            (item["onnx"], item["ncnn"], tuple(item["onnx_shape"])) for item in self.tpl_io["outputs"]
        ]

        search_input_items = [item for item in self.srch_io["inputs"] if item["onnx"] == "search"]
        if len(search_input_items) != 1:
            raise RuntimeError(f"Expected one 'search' input in {srch_io_path}")
        self._search_blob = search_input_items[0]["ncnn"]
        self._search_shape = tuple(search_input_items[0]["ncnn_shape"])
        self._cache_input_specs = [
            (item["onnx"], item["ncnn"], tuple(item["ncnn_shape"]))
            for item in self.srch_io["inputs"]
            if item["onnx"] != "search"
        ]
        self._srch_outputs = {
            item["onnx"]: (item["ncnn"], tuple(item["onnx_shape"])) for item in self.srch_io["outputs"]
        }
        for key in ("score", "size", "offset"):
            if key not in self._srch_outputs:
                raise RuntimeError(f"Missing output {key!r} in {srch_io_path}")

        self.cached_inputs: Optional[List[Tuple[str, np.ndarray]]] = None
        prec = "FP16" if fp16 else "FP32"
        print(
            f"[TaCo/ncnn] dir={model_dir} | precision={prec} | threads={threads} | "
            f"setting={setting} | caches={len(self._cache_input_specs)}"
        )

    def initialize(self, image_rgb: np.ndarray, info: dict) -> float:
        """Compute and cache template operands once on frame 0; returns graph time (ms)."""
        z_tensor, tbb_xyxy = self._prepare_template(image_rgb, info["init_bbox"])
        t0 = time.perf_counter()
        feeds = {"z": z_tensor, **compute_template_geometry(tbb_xyxy)}
        ex = self.tpl_net.create_extractor()
        for onnx_name, blob, shape in self._tpl_inputs:
            arr = np.ascontiguousarray(feeds[onnx_name], dtype=np.float32).reshape(shape)
            ex.input(blob, self._ncnn.Mat(arr))

        raw_caches: Dict[str, np.ndarray] = {}
        for onnx_name, blob, shape in self._tpl_outputs:
            ret, mat = ex.extract(blob)
            if ret != 0:
                raise RuntimeError(f"ncnn template extract failed on {blob} ({onnx_name}), rc={ret}")
            raw_caches[onnx_name] = np.array(mat, dtype=np.float32).reshape(shape)

        # Pre-reshape cached operands once into the search graph's expected ncnn shapes.
        self.cached_inputs = [
            (blob, np.ascontiguousarray(raw_caches[onnx_name], dtype=np.float32).reshape(shape))
            for onnx_name, blob, shape in self._cache_input_specs
        ]
        return (time.perf_counter() - t0) * 1000.0

    def track(self, image_rgb: np.ndarray) -> dict:
        """Run one search frame using cached template operands."""
        if self.cached_inputs is None:
            raise RuntimeError("Call initialize() before track().")
        height, width = image_rgb.shape[:2]
        x_tensor, resize_factor = self._prepare_search(image_rgb)

        t0 = time.perf_counter()
        ex = self.srch_net.create_extractor()
        x_ncnn = np.ascontiguousarray(x_tensor, dtype=np.float32).reshape(self._search_shape)
        ex.input(self._search_blob, self._ncnn.Mat(x_ncnn))
        for blob, arr in self.cached_inputs:
            ex.input(blob, self._ncnn.Mat(arr))

        outputs: Dict[str, np.ndarray] = {}
        for name in ("score", "size", "offset"):
            blob, shape = self._srch_outputs[name]
            ret, mat = ex.extract(blob)
            if ret != 0:
                raise RuntimeError(f"ncnn search extract failed on {blob} ({name}), rc={ret}")
            outputs[name] = np.array(mat, dtype=np.float32).reshape(shape)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        res = self._postprocess(
            outputs["score"], outputs["size"], outputs["offset"], resize_factor, height, width
        )
        res["infer_ms"] = infer_ms
        return res


def _parse_bbox(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"--init-bbox must have 4 values 'x,y,w,h', got: {raw!r}")
    bbox = [float(v) for v in parts]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Bounding box width and height must be positive, got: {bbox}")
    return bbox


def _iter_frames(source: str) -> Tuple[Iterator[np.ndarray], float, Optional[int]]:
    """Yield BGR frames from a video file, image directory, or integer camera index."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0

        def _gen_cam():
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame
            finally:
                cap.release()

        return _gen_cam(), fps, None

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
                    raise RuntimeError(f"Failed to read image: {path}")
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


def _draw_overlay(
    frame_bgr: np.ndarray, bbox: List[float], score: float, infer_fps: float, frame_idx: int
) -> np.ndarray:
    vis = frame_bgr.copy()
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
    label = f"Pi5 ncnn #{frame_idx}  score={score:.2f}  {infer_fps:.1f} FPS"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TaCo ncnn video / image-folder / camera tracking on Raspberry Pi 5."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="ncnn export directory containing template/ and search/ (e.g., build/ncnn/TaCo-T).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video file, directory of frames, or camera index (e.g., 0).",
    )
    parser.add_argument("--init-bbox", type=str, default=None, help="Initial bbox 'x,y,w,h' on frame 0.")
    parser.add_argument("--output", type=str, default=None, help="Optional annotated output video path (.mp4).")
    parser.add_argument("--save-bbox", type=str, default=None, help="Optional CSV path to save per-frame boxes.")
    parser.add_argument("--threads", type=int, default=4, help="ncnn CPU threads (default: 4 on Pi 5).")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 and run ncnn in FP32.")
    parser.add_argument(
        "--setting",
        type=str,
        default="export_default",
        choices=list(NCNN_SETTINGS.keys()),
        help="ncnn runtime kernel option (default: export_default; try winograd_f23 for TaCo-T).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--show", action="store_true", help="Display live window (press 'q' or ESC to quit).")
    args = parser.parse_args()

    # Pin to Cortex-A76 cores 0..threads-1 on Linux (matches paper's Pi 5 protocol)
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(range(args.threads)))
        except OSError:
            pass

    frames_iter, video_fps, total_frames = _iter_frames(args.input)
    try:
        first_bgr = next(frames_iter)
    except StopIteration:
        print("Input contains zero frames.", file=sys.stderr)
        sys.exit(1)

    if args.init_bbox is not None:
        init_bbox = _parse_bbox(args.init_bbox)
    else:
        roi = cv2.selectROI("Select Target (TaCo Pi 5)", first_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target (TaCo Pi 5)")
        init_bbox = [float(v) for v in roi]
        if init_bbox[2] <= 0 or init_bbox[3] <= 0:
            print("No target selected; exiting.", file=sys.stderr)
            sys.exit(1)

    tracker = TaCoNCNNTracker(
        args.model_dir,
        threads=args.threads,
        fp16=not args.fp32,
        setting=args.setting,
    )

    writer = None
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        h, w = first_bgr.shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    tpl_graph_ms = tracker.initialize(first_rgb, {"init_bbox": init_bbox})
    init_total_ms = (time.perf_counter() - t0) * 1000.0

    boxes: List[List[float]] = [init_bbox]
    total_ms_list: List[float] = []
    infer_ms_list: List[float] = []

    if writer is not None or args.show:
        vis = _draw_overlay(first_bgr, init_bbox, 1.0, 1000.0 / max(tpl_graph_ms, 1e-3), 0)
        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("TaCo Pi 5 (ncnn)", vis)

    frame_idx = 0
    try:
        for frame_bgr in frames_iter:
            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            out = tracker.track(frame_rgb)
            total_ms = (time.perf_counter() - t0) * 1000.0
            infer_ms = out["infer_ms"]

            total_ms_list.append(total_ms)
            infer_ms_list.append(infer_ms)
            boxes.append(out["target_bbox"])

            if writer is not None or args.show:
                vis = _draw_overlay(
                    frame_bgr, out["target_bbox"], out["best_score"], 1000.0 / max(infer_ms, 1e-3), frame_idx
                )
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("TaCo Pi 5 (ncnn)", vis)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
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

    mean_infer_ms = (sum(infer_ms_list) / len(infer_ms_list)) if infer_ms_list else 0.0
    mean_total_ms = (sum(total_ms_list) / len(total_ms_list)) if total_ms_list else 0.0
    infer_fps = 1000.0 / mean_infer_ms if mean_infer_ms > 0 else 0.0
    total_fps = 1000.0 / mean_total_ms if mean_total_ms > 0 else 0.0
    total_str = f"/{total_frames}" if total_frames is not None else ""
    print(
        f"Tracked {len(boxes)}{total_str} frames | "
        f"template: {tpl_graph_ms:.2f} ms graph ({init_total_ms:.2f} ms total) | "
        f"search graph: {mean_infer_ms:.2f} ms ({infer_fps:.1f} FPS) | "
        f"end-to-end: {mean_total_ms:.2f} ms ({total_fps:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
