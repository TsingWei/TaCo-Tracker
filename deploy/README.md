# Edge deployment guide

This directory converts trained TaCo checkpoints into two-branch deployable graphs (`template` + `search`) for **ONNX Runtime**, **Raspberry Pi 5 (ncnn FP16)**, and **Rockchip RV1126B (RKNN FP16)** with automated numerical verification.

| File | Role |
| --- | --- |
| `export_onnx.py` | Reparameterize PyTorch weights and export `template.onnx` + `search.onnx` with ORT FP32 verification. |
| `export_ncnn.py` | Lower template `RoiAlign` to `MatMul`, convert via PNNX to ncnn, generate `io_map.json`, and verify FP32 / FP16 outputs. |
| `export_rknn.py` | Lower template `RoiAlign`, split the FP32 FiLM head (`film_head.npz`), and compile RKNN FP16 models for RV1126B. |
| `pi5_demo_video.py` | PyTorch-free Raspberry Pi 5 (ncnn FP16/FP32) video / image-folder / camera tracking demo. |
| `rv1126b_demo_video.py` | PyTorch-free Rockchip RV1126B NPU (RKNN FP16 via `librknnrt.so`) video / image-folder / camera tracking demo. |
| `template_lowering.py` | Host-side box geometry (`compute_template_geometry`), `RoiAlign -> MatMul` ONNX lowering, and RKNN FiLM-head split utilities. |
| `verify_utils.py` | Shared ONNX Runtime FP32 reference generation and numerical verification gates. |

## 1. Export ONNX

Run from the repository root with the full environment (`requirements_train.txt`):

```bash
python deploy/export_onnx.py --model TaCo-T --weights weights/TaCo-T.pth
```

The script applies deploy-time reparameterization, writes `onnx_models/TaCo-T/template.onnx` and `search.onnx`, and verifies the chained ONNX Runtime FP32 outputs against PyTorch FP32 (relative L2 ≤ 10⁻⁴, identical peak cell, box diff ≤ 10⁻⁵).

## 2. Raspberry Pi 5 (Arm CPU / ncnn)

### Conversion (on host)

Install `pnnx` and Python `ncnn` on the conversion host, then export both graphs:

```bash
python deploy/export_ncnn.py --model TaCo-T --out build/ncnn/TaCo-T --graph both --check-fp16
```

`export_ncnn.py` lowers template `RoiAlign` into a standard `MatMul` backed by host-computed box geometry, so every layer maps natively to ncnn (83 layers for `TaCo-T` template, 74 layers for `TaCo-T` search) with zero unsupported ops.

### Real-video inference (on Raspberry Pi 5)

Copy `build/ncnn/TaCo-T/`, `deploy/`, `benchmark/`, and `tracker.py` to the Raspberry Pi 5. Only `ncnn`, `numpy`, and `opencv-python` are required on the board (no PyTorch or ONNX Runtime):

```bash
pip install ncnn numpy opencv-python

python deploy/pi5_demo_video.py \
  --model-dir build/ncnn/TaCo-T \
  --input sample.mp4 \
  --init-bbox 241,307,308,153 \
  --output tracked_pi5.mp4 \
  --threads 4 \
  --setting winograd_f23
```

Pass `--input 0 --show` for live USB/CSI camera tracking, or omit `--init-bbox` to draw the initial box interactively.

### Raspberry Pi 5 real-video verification

Measured on a **Raspberry Pi 5 (Arm Cortex-A76, 4 threads, ncnn FP16, `--setting winograd_f23`)** running `deploy/pi5_demo_video.py` on a 120-frame `1280×720` clip from LaSOT (`car-1`), with zero PyTorch or ONNX Runtime installed on the device:

![TaCo-T real-video tracking on Raspberry Pi 5 with ncnn FP16](assets/pi5_ncnn_demo.jpg)

*Frame 119 of the 720p `car-1` sequence tracked on Raspberry Pi 5 with `TaCo-T` (ncnn FP16, 4 threads). The green box and top-left HUD are produced directly by `deploy/pi5_demo_video.py`; the thin white box overlays the dataset ground truth for reference.*

| Metric | Raspberry Pi 5 measured (`TaCo-T`, FP16, 4 threads) |
| --- | --- |
| Template initialization (frame 0 only) | **8.77 ms** graph (15.91 ms incl. crop & preprocess) |
| Search graph latency / throughput | **10.39 ms** (**96.2 FPS**) |
| End-to-end tracking (crop + ncnn + box decode) | **13.83 ms** (**72.3 FPS**) |
| Tracking accuracy vs. GT (120 frames) | **Mean IoU 0.8769** · Min IoU 0.8453 · **100.0%** frames with IoU > 0.5 |

> Note: Unlike the warmed-up standalone benchmarks in `benchmark/pi5_benchmark.py` (reported in the main [Model Zoo](../README.md#taco-model-zoo) table), the frame-0 template measurement above is an un-warmed cold start that also includes reshaping and caching all template operands for the search session.

### Latency & numerical benchmark (on Raspberry Pi 5)

```bash
python benchmark/pi5_benchmark.py --model-dir build/ncnn/TaCo-T/search \
  --graph search --threads 4 --screen --out results/pi5_t_search.json
python benchmark/pi5_benchmark.py --model-dir build/ncnn/TaCo-T/template \
  --graph template --threads 4 --out results/pi5_t_template.json
```

## 3. Rockchip RV1126B (NPU / RKNN)

### Conversion (on x86-64 Linux host)

With RKNN Toolkit2 v2.3.2 installed:

```bash
python deploy/export_rknn.py --model TaCo-T --out build/rknn/TaCo-T --graph both --verify-sim
```

The search graph compiles directly to pure FP16 RKNN (`search.rknn`) with zero CPU fallback. The template graph lowers `RoiAlign` to `MatMul` and partitions the lightweight FiLM MLP head into `film_head.npz` (host CPU FP32) alongside `template_npu.rknn` (NPU FP16).

### Real-video inference (on RV1126B)

Copy `build/rknn/TaCo-T/`, `deploy/`, `benchmark/rknn_runtime.py`, and `tracker.py` to the RV1126B board. Only RKNN Runtime v2.3.2 (`/usr/lib/librknnrt.so`), `numpy`, and `opencv-python` are required on the board (no PyTorch or ONNX Runtime):

```bash
python deploy/rv1126b_demo_video.py \
  --model-dir build/rknn/TaCo-T \
  --input sample.mp4 \
  --init-bbox 241,307,308,153 \
  --output tracked_rv1126b.mp4 \
  --save-bbox boxes_rv1126b.csv
```

Pass `--rknn-lib /path/to/librknnrt.so` if the RKNN shared library is in a custom location, `--disable-cpuidle` (root) to minimize NPU dispatch jitter, or `--input ./frames` for an image directory.

### Rockchip RV1126B real-video verification

Measured on a **Rockchip RV1126B NPU (950 MHz, RKNN Runtime v2.3.2, FP16, native NHWC FP16 `pass_through=1` input, zero CPU fallback operators `0/61`)** running `deploy/rv1126b_demo_video.py` on the same 120-frame `1280×720` clip from LaSOT (`car-1`), with zero PyTorch or ONNX Runtime installed on the board:

![TaCo-T real-video tracking on Rockchip RV1126B NPU with RKNN FP16](assets/rv1126b_rknn_demo.jpg)

*Frame 119 of the 720p `car-1` sequence tracked on Rockchip RV1126B NPU with `TaCo-T` (RKNN FP16). The green box and top-left HUD are produced directly by `deploy/rv1126b_demo_video.py`; the thin white box overlays the dataset ground truth for reference.*

| Metric | Rockchip RV1126B measured (`TaCo-T`, RKNN FP16) |
| --- | --- |
| Template initialization (frame 0 only, hybrid NPU + CPU FP32 FiLM + cache upload) | **34.58 ms** (39.22 ms incl. crop & preprocess) |
| Search NPU execution (`rknn_run`) | **5.66 ms** (**176.6 FPS**) · `0/61` CPU fallback ops |
| Search NPU I/O + run (`set_input` + `rknn_run` + `get_outputs`) | **6.96 ms** (**143.6 FPS**) |
| End-to-end Python tracking (crop + LUT norm + NPU + box decode) | **18.76 ms** (**53.3 FPS**) |
| Tracking accuracy vs. GT (120 frames) | **Mean IoU 0.8760** · Min IoU 0.8507 · **100.0%** frames with IoU > 0.5 |

> Note: The video demo measures an un-warmed frame-0 template cold start (including cache upload to the search graph) and per-frame Python `ctypes` calls, whereas the main [Model Zoo](../README.md#taco-model-zoo) table reports warmed-up steady-state `benchmark/rv1126b_benchmark.py` timings (C `rknn_benchmark` median of five 1,000-call rounds for search).

### Latency & numerical benchmark (on RV1126B)

Copy `build/rknn/TaCo-T/`, `benchmark/`, and `deploy/` to the board (RKNN Runtime v2.3.2 `/usr/lib/librknnrt.so` + `numpy`):

```bash
python benchmark/rv1126b_benchmark.py --model-dir build/rknn/TaCo-T/search \
  --graph search --cbench /path/to/rknn_benchmark_aarch64 --disable-cpuidle \
  --out results/rv1126b_t_search.json
python benchmark/rv1126b_benchmark.py --model-dir build/rknn/TaCo-T/template \
  --graph template --warmup 20 --disable-cpuidle --out results/rv1126b_t_template.json
```

## Numerical verification gates

All edge conversions and benchmarks validate against ONNX Runtime FP32 references (`deploy/verify_utils.py`):

- **Search graph:** all outputs finite, relative L2 ≤ 0.02, max absolute error ≤ max(0.005, 0.05 × max|ref|), identical score-peak cell, and decoded-box IoU ≥ 0.98 (plus zero CPU fallback operators on RKNN).
- **Template graph:** all cached operands finite with relative L2 ≤ 0.02.
