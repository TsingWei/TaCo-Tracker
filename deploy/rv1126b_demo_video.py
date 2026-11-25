#!/usr/bin/env python3
"""Rockchip RV1126B NPU video / image-folder / camera inference with RKNN Runtime.

Runs the two-branch RKNN FP16 graphs exported by ``deploy/export_rknn.py`` directly
on a Rockchip RV1126B board via ``librknnrt.so`` (ctypes) with zero PyTorch or
ONNX Runtime dependency:
  - Frame 0 (template): hybrid pipeline = host box geometry + NPU FP16 template
    backbone & kernel generation (``template_npu.rknn``) + host CPU FP32 FiLM head
    (``film_head.npz``). All cached operands are uploaded to ``search.rknn`` once.
  - Subsequent frames (search): pure NPU FP16 graph (``search.rknn``) updating only
    the search crop tensor (using native NHWC FP16 ``pass_through=1`` when supported
    by the compiled RKNN model, with automatic fallback to standard FP32 inputs).

Step 1 (on x86-64 Linux conversion host with rknn-toolkit2 v2.3.2):
    python deploy/export_onnx.py --model TaCo-T --weights weights/TaCo-T.pth
    python deploy/export_rknn.py --model TaCo-T --out build/rknn/TaCo-T --graph both --verify-sim

Step 2 (copy build/rknn/TaCo-T/, deploy/, benchmark/rknn_runtime.py, and tracker.py to RV1126B):
    # Requires /usr/lib/librknnrt.so (RKNN Runtime v2.3.2), numpy, and opencv-python
    python deploy/rv1126b_demo_video.py \
        --model-dir build/rknn/TaCo-T \
        --input sample.mp4 \
        --init-bbox 120,80,96,64 \
        --output tracked_rv1126b.mp4

    # Optional: pass --disable-cpuidle for lowest NPU dispatch jitter (requires root)
    # Optional: pass --input 0 --show for live camera tracking (when display is available)
"""

import argparse
import contextlib
import ctypes as c
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
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from rknn_runtime import RknnHybridTemplateRunner, RknnInput, RknnRunner, RknnTensorAttr, _check
from tracker import _BaseTaCoTracker, sample_target

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CPU_ROOT = Path("/sys/devices/system/cpu")
RKNN_QUERY_NATIVE_INPUT_ATTR = 8
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_NHWC = 1


def _to_rknn_input(arr: np.ndarray) -> np.ndarray:
    """Convert 4-D NCHW arrays to NHWC for librknnrt; lower-rank arrays keep their shape."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        return np.ascontiguousarray(arr.transpose(0, 2, 3, 1), dtype=np.float32)
    return arr


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


class TaCoRKNNTracker(_BaseTaCoTracker):
    """PyTorch-free TaCo tracker backed by Rockchip RKNN Runtime for RV1126B NPU."""

    def __init__(self, model_dir: Path, lib_path: str = "/usr/lib/librknnrt.so"):
        super().__init__(
            {
                "template_size": 128,
                "search_size": 256,
                "template_factor": 2.0,
                "search_factor": 4.0,
                "stride": 16,
            }
        )
        tpl_dir = model_dir / "template"
        srch_dir = model_dir / "search"

        tpl_manifest_path = tpl_dir / "manifest.json"
        srch_manifest_path = srch_dir / "manifest.json"
        if not tpl_manifest_path.is_file() or not srch_manifest_path.is_file():
            raise FileNotFoundError(
                f"Expected template/manifest.json and search/manifest.json under {model_dir}. "
                "Export both graphs with: python deploy/export_rknn.py --model <variant> --out <dir> --graph both"
            )

        self.srch_manifest = json.loads(srch_manifest_path.read_text())
        srch_rknn_path = srch_dir / self.srch_manifest["rknn"]
        if not srch_rknn_path.is_file():
            raise FileNotFoundError(f"Missing RKNN search model: {srch_rknn_path}")

        self.tpl_runner: Optional[RknnHybridTemplateRunner] = None
        self.srch_runner: Optional[RknnRunner] = None
        try:
            self.tpl_runner = RknnHybridTemplateRunner(tpl_dir, lib_path=lib_path)
            self.srch_runner = RknnRunner(srch_rknn_path, lib_path=lib_path, collect_perf=False)
        except BaseException:
            self.close()
            raise

        # Resolve exact search input ordering (handles arbitrary tensor order from RKNN metadata)
        manifest_in_names = list(self.srch_manifest.get("input_names", []))
        if manifest_in_names and set(self.srch_runner.in_names) == set(manifest_in_names):
            self._srch_in_names = list(self.srch_runner.in_names)
        elif "search" in self.srch_runner.in_names:
            self._srch_in_names = list(self.srch_runner.in_names)
        elif len(manifest_in_names) == self.srch_runner.n_in and "search" in manifest_in_names:
            self._srch_in_names = manifest_in_names
        else:
            self.close()
            raise RuntimeError(
                f"Cannot match search RKNN inputs {self.srch_runner.in_names} with manifest {manifest_in_names}"
            )
        self._search_idx = self._srch_in_names.index("search")

        # Resolve search output indices for (score, size, offset)
        manifest_out_names = list(self.srch_manifest.get("output_names", ["score", "size", "offset"]))
        if {"score", "size", "offset"}.issubset(self.srch_runner.out_names):
            self._out_idx = {k: self.srch_runner.out_names.index(k) for k in ("score", "size", "offset")}
        elif {"score", "size", "offset"}.issubset(manifest_out_names) and len(manifest_out_names) == self.srch_runner.n_out:
            self._out_idx = {k: manifest_out_names.index(k) for k in ("score", "size", "offset")}
        else:
            self.close()
            raise RuntimeError(
                f"Cannot match search RKNN outputs {self.srch_runner.out_names} with manifest {manifest_out_names}"
            )

        # Check whether the search input accepts native NHWC FP16 pass_through=1
        # (eliminates ~12 ms/frame CPU layout conversion on RV1126B while staying bit-exact).
        self._fast_fp16_passthrough = self._probe_native_fp16_nhwc()
        self._single_inp = (RknnInput * 1)()

        # Precompute ImageNet normalization LUTs for direct uint8 NHWC -> FP16/FP32 normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        vals = (np.arange(256, dtype=np.float32)[:, None] / 255.0 - mean[None, :]) / std[None, :]
        self._norm_lut_f16 = [np.ascontiguousarray(vals[:, ch], dtype=np.float16) for ch in range(3)]
        self._norm_lut_f32 = [np.ascontiguousarray(vals[:, ch], dtype=np.float32) for ch in range(3)]
        self._search_buf_f16 = np.empty((1, self.search_size, self.search_size, 3), dtype=np.float16)
        self._search_buf_f32 = np.empty((1, self.search_size, self.search_size, 3), dtype=np.float32)

        self._cached_rknn_inputs: Optional[Dict[str, np.ndarray]] = None
        n_caches = len(self._srch_in_names) - 1
        platform = self.srch_manifest.get("target_platform", "rv1126b")
        dtype = self.srch_manifest.get("float_dtype", "float16")
        mode = "fp16-passthrough" if self._fast_fp16_passthrough else "fp32-standard"
        print(
            f"[TaCo/RKNN] dir={model_dir} | target={platform} | dtype={dtype} | "
            f"input_mode={mode} | caches={n_caches} | lib={lib_path}"
        )

    def _probe_native_fp16_nhwc(self) -> bool:
        runner = self.srch_runner
        if runner is None or not hasattr(runner, "lib") or not hasattr(runner, "ctx"):
            return False
        expected_bytes = self.search_size * self.search_size * 3 * 2
        attr = RknnTensorAttr()
        attr.index = self._search_idx
        ret = runner.lib.rknn_query(runner.ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, c.byref(attr), c.sizeof(attr))
        if ret != 0:
            return False
        dims = tuple(int(attr.dims[d]) for d in range(attr.n_dims))
        return (
            dims == (1, self.search_size, self.search_size, 3)
            and int(attr.fmt) == RKNN_TENSOR_NHWC
            and int(attr.type) == RKNN_TENSOR_FLOAT16
            and int(attr.size_with_stride) == expected_bytes
        )

    def close(self) -> None:
        if self.srch_runner is not None:
            self.srch_runner.close()
            self.srch_runner = None
        if self.tpl_runner is not None:
            self.tpl_runner.close()
            self.tpl_runner = None

    def __enter__(self) -> "TaCoRKNNTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self, image_rgb: np.ndarray, info: dict) -> float:
        """Run hybrid NPU+CPU template stage once on frame 0; returns template execution time (ms)."""
        if self.tpl_runner is None or self.srch_runner is None:
            raise RuntimeError("Tracker has already been closed.")
        z_tensor, tbb_xyxy = self._prepare_template(image_rgb, info["init_bbox"])

        t0 = time.perf_counter()
        raw_caches = self.tpl_runner.run(z_tensor, tbb_xyxy)
        # Pre-format cached operands once into the NHWC / contiguous FP32 layout expected by librknnrt.
        cached: Dict[str, np.ndarray] = {}
        for name in self._srch_in_names:
            if name == "search":
                continue
            if name not in raw_caches:
                raise KeyError(f"Template runner output missing search input {name!r}")
            cached[name] = _to_rknn_input(raw_caches[name])
        self._cached_rknn_inputs = cached

        # Upload cached template operands into the search NPU context once per video
        # when using single-input FP16 pass-through updates in track().
        if self._fast_fp16_passthrough:
            dummy_search = np.zeros((1, self.search_size, self.search_size, 3), dtype=np.float32)
            init_feeds = [dummy_search if name == "search" else cached[name] for name in self._srch_in_names]
            self.srch_runner.set_inputs(init_feeds)

        return (time.perf_counter() - t0) * 1000.0

    def _prepare_search_nhwc(self, image_rgb: np.ndarray, fp16: bool) -> Tuple[np.ndarray, float]:
        if self.state is None:
            raise RuntimeError("Tracker must be initialized with initialize() before calling track().")
        self.frame_id += 1
        x_patch, resize_factor = sample_target(image_rgb, self.state, self.search_factor, self.search_size)
        if fp16:
            buf = self._search_buf_f16
            luts = self._norm_lut_f16
        else:
            buf = self._search_buf_f32
            luts = self._norm_lut_f32
        buf[0, :, :, 0] = luts[0][x_patch[:, :, 0]]
        buf[0, :, :, 1] = luts[1][x_patch[:, :, 1]]
        buf[0, :, :, 2] = luts[2][x_patch[:, :, 2]]
        return buf, resize_factor

    def track(self, image_rgb: np.ndarray) -> dict:
        """Run one search frame on the RV1126B NPU using cached template operands."""
        if self.srch_runner is None:
            raise RuntimeError("Tracker has already been closed.")
        if self._cached_rknn_inputs is None:
            raise RuntimeError("Call initialize() before track().")
        height, width = image_rgb.shape[:2]

        if self._fast_fp16_passthrough:
            x_nhwc, resize_factor = self._prepare_search_nhwc(image_rgb, fp16=True)
            self._single_inp[0] = RknnInput(
                self._search_idx,
                x_nhwc.ctypes.data,
                x_nhwc.nbytes,
                1,
                RKNN_TENSOR_FLOAT16,
                RKNN_TENSOR_NHWC,
            )
            t0 = time.perf_counter()
            _check(self.srch_runner.lib.rknn_inputs_set(self.srch_runner.ctx, 1, self._single_inp), "rknn_inputs_set")
            t_run0 = time.perf_counter()
            self.srch_runner.run_only()
            npu_ms = (time.perf_counter() - t_run0) * 1000.0
            raw_outs = self.srch_runner.get_outputs()
            infer_ms = (time.perf_counter() - t0) * 1000.0
        else:
            x_nhwc, resize_factor = self._prepare_search_nhwc(image_rgb, fp16=False)
            feeds = [x_nhwc if name == "search" else self._cached_rknn_inputs[name] for name in self._srch_in_names]
            t0 = time.perf_counter()
            raw_outs = self.srch_runner.run(feeds)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            npu_ms = infer_ms

        score = np.asarray(raw_outs[self._out_idx["score"]], dtype=np.float32).reshape(
            1, 1, self.feat_sz, self.feat_sz
        )
        size = np.asarray(raw_outs[self._out_idx["size"]], dtype=np.float32).reshape(
            1, 2, self.feat_sz, self.feat_sz
        )
        offset = np.asarray(raw_outs[self._out_idx["offset"]], dtype=np.float32).reshape(
            1, 2, self.feat_sz, self.feat_sz
        )

        res = self._postprocess(score, size, offset, resize_factor, height, width)
        res["infer_ms"] = infer_ms
        res["npu_ms"] = npu_ms
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
    label = f"RV1126B NPU #{frame_idx}  score={score:.2f}  {infer_fps:.1f} FPS"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TaCo RKNN FP16 video / image-folder / camera tracking on Rockchip RV1126B NPU."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="RKNN export directory containing template/ and search/ (e.g., build/rknn/TaCo-T).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video file, directory of frames, or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--init-bbox",
        type=str,
        default=None,
        help="Initial bbox 'x,y,w,h' on frame 0 (recommended on headless RV1126B boards).",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional annotated output video path (.mp4).")
    parser.add_argument("--save-bbox", type=str, default=None, help="Optional CSV path to save per-frame boxes.")
    parser.add_argument(
        "--rknn-lib",
        type=str,
        default="/usr/lib/librknnrt.so",
        help="Path to Rockchip RKNN runtime library on the board (default: /usr/lib/librknnrt.so).",
    )
    parser.add_argument(
        "--disable-cpuidle",
        action="store_true",
        help="Temporarily disable CPU state1 deep idle during tracking (requires root).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--show", action="store_true", help="Display live window (press 'q' or ESC to quit).")
    args = parser.parse_args()

    frames_iter, video_fps, total_frames = _iter_frames(args.input)
    try:
        first_bgr = next(frames_iter)
    except StopIteration:
        print("Input contains zero frames.", file=sys.stderr)
        sys.exit(1)

    if args.init_bbox is not None:
        init_bbox = _parse_bbox(args.init_bbox)
    else:
        try:
            roi = cv2.selectROI("Select Target (TaCo RV1126B)", first_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Target (TaCo RV1126B)")
        except cv2.error as exc:
            print(
                f"Interactive ROI selection unavailable ({exc}). Pass --init-bbox x,y,w,h on headless boards.",
                file=sys.stderr,
            )
            sys.exit(1)
        init_bbox = [float(v) for v in roi]
        if init_bbox[2] <= 0 or init_bbox[3] <= 0:
            print("No target selected; exiting.", file=sys.stderr)
            sys.exit(1)

    writer = None
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        h, w = first_bgr.shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

    boxes: List[List[float]] = [init_bbox]
    total_ms_list: List[float] = []
    infer_ms_list: List[float] = []
    npu_ms_list: List[float] = []

    with _maybe_disable_cpuidle(args.disable_cpuidle), TaCoRKNNTracker(args.model_dir, lib_path=args.rknn_lib) as tracker:
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        tpl_graph_ms = tracker.initialize(first_rgb, {"init_bbox": init_bbox})
        init_total_ms = (time.perf_counter() - t0) * 1000.0

        if writer is not None or args.show:
            vis = _draw_overlay(first_bgr, init_bbox, 1.0, 1000.0 / max(tpl_graph_ms, 1e-3), 0)
            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow("TaCo RV1126B (RKNN NPU)", vis)

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
                npu_ms = out["npu_ms"]

                total_ms_list.append(total_ms)
                infer_ms_list.append(infer_ms)
                npu_ms_list.append(npu_ms)
                boxes.append(out["target_bbox"])

                if writer is not None or args.show:
                    vis = _draw_overlay(
                        frame_bgr, out["target_bbox"], out["best_score"], 1000.0 / max(infer_ms, 1e-3), frame_idx
                    )
                    if writer is not None:
                        writer.write(vis)
                    if args.show:
                        cv2.imshow("TaCo RV1126B (RKNN NPU)", vis)
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

    mean_npu_ms = (sum(npu_ms_list) / len(npu_ms_list)) if npu_ms_list else 0.0
    mean_infer_ms = (sum(infer_ms_list) / len(infer_ms_list)) if infer_ms_list else 0.0
    mean_total_ms = (sum(total_ms_list) / len(total_ms_list)) if total_ms_list else 0.0
    npu_fps = 1000.0 / mean_npu_ms if mean_npu_ms > 0 else 0.0
    infer_fps = 1000.0 / mean_infer_ms if mean_infer_ms > 0 else 0.0
    total_fps = 1000.0 / mean_total_ms if mean_total_ms > 0 else 0.0
    total_str = f"/{total_frames}" if total_frames is not None else ""
    print(
        f"Tracked {len(boxes)}{total_str} frames | "
        f"template (hybrid): {tpl_graph_ms:.2f} ms ({init_total_ms:.2f} ms total) | "
        f"search NPU run: {mean_npu_ms:.2f} ms ({npu_fps:.1f} FPS) | "
        f"NPU I/O+run: {mean_infer_ms:.2f} ms ({infer_fps:.1f} FPS) | "
        f"end-to-end: {mean_total_ms:.2f} ms ({total_fps:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
