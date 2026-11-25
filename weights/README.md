# Trained TaCo checkpoints

## Mixed training (epoch 300)

The mixed-training epoch-300 checkpoints below correspond to the [main benchmark results](../README.md#taco-model-zoo). Each file stores the training-form model state dict, with FP16 storage for weight tensors of rank two and above and FP32 storage for normalization, bias, and floating-point buffers.

| File | Size (MB) | Exact size (bytes) | SHA-256 | Google Drive |
|---|---:|---:|---|---|
| `TaCo-T.pth` | 11.3 | 11,309,119 | `dfd15e668d6f9ae453f37bbe467b9bac81f3b6ceefbcdb3bba91da11e6883bc6` | [Download](https://drive.google.com/file/d/1ssVZSydXKStdm0BhkqFnXYZAoE0JfIyB/view?usp=drivesdk) |
| `TaCo-S.pth` | 14.2 | 14,195,070 | `cdcf9c30d85d029297954f349b3a6b70d66cc6ef27afe96007d5aef08d570064` | [Download](https://drive.google.com/file/d/1ujRaXlVuCWvOUoV-zaMM8opKPOWI9UAH/view?usp=drivesdk) |
| `TaCo-B.pth` | 24.9 | 24,915,542 | `22dc7a2a3f5625919f503981a2b4bbf4ab300fb5f4bdfe7b095e02fb58088e8d` | [Download](https://drive.google.com/file/d/105bSFs7RoP1z7DsPvT18uKwS2o3wlBDH/view?usp=drivesdk) |

Save the downloaded files in `weights/` with the filenames above for default model lookup. From the repository root, verify a download with `shasum -a 256 weights/TaCo-T.pth` (or `sha256sum weights/TaCo-T.pth` on Linux).

Pass an explicit checkpoint to PyTorch evaluation, the demo, or ONNX export when using a different path. Match the model name to the checkpoint variant.

```bash
python eval.py TaCo-T --dataset lasot --backend pytorch --device cuda --fp16 \
  --data-root ./data/lasot --weights /path/to/trained_checkpoint.pth
python deploy/export_onnx.py --model TaCo-T --weights /path/to/trained_checkpoint.pth
```

ONNX Runtime reads the exported template and search graphs under `onnx_models/TaCo-T/`. Export your checkpoint before using it with that backend. For training initialization, see the [ImageNet backbone guide](../pretrained_models/README.md).

## GOT-10k-only (epoch 100)

These model state dicts reproduce the [GOT-10k one-shot evaluation](../README.md#download-predictions). Pass the matching filename explicitly with `--weights`.

| File | Size (MB) | Exact size (bytes) | SHA-256 | Google Drive |
|---|---:|---:|---|---|
| `TaCo-T-GOT10kOnly-E100.pth` | 42.6 | 42,622,067 | `36d007065d1f8f6039779450438e97ca2853acd1112505298ea0fe0a0f1f3cad` | [Download](https://drive.google.com/file/d/1_OdXsDuHB7tcpbeY5KXMv38DsF8sHXS9/view?usp=drivesdk) |
| `TaCo-S-GOT10kOnly-E100.pth` | 41.6 | 41,636,212 | `b9b725a814bc225bdddae0b4c3aa665377439f9738166357a9e12ca442bbcec5` | [Download](https://drive.google.com/file/d/1uQ86lwxQhgzHovzloZM5KUbnWsOjnDd_/view?usp=drivesdk) |
| `TaCo-B-GOT10kOnly-E100.pth` | 71.7 | 71,698,702 | `3c18165dd60cb3a0445fb325f424ae3f2445ae8fdecdde2882049230a2bc46cf` | [Download](https://drive.google.com/file/d/1ILCldviF-36d0H0jPLEhSrEk6_UZ0BlW/view?usp=drivesdk) |
