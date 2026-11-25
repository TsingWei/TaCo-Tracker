#!/usr/bin/env python3
"""Export PyTorch TaCo weights into two-branch ONNX graphs (template.onnx + search.onnx).

Loads the bundled FP16-stored PyTorch weights in ``weights/<model>.pth`` (casting
weights to FP32), performs deploy-time reparameterization (RepConv head + RepViT
backbone for TaCo-S / TaCo-B), exports both ``template.onnx`` and ``search.onnx``
into ``onnx_models/<model>/``, and verifies the exported ONNX chain against PyTorch
using ONNX Runtime FP32 (relative L2 <= 1e-4, identical score-peak cell, box diff <= 1e-5).

Examples:
    # Export all three variants using the bundled weights/TaCo-*.pth
    python deploy/export_onnx.py --model TaCo-T
    python deploy/export_onnx.py --model TaCo-S
    python deploy/export_onnx.py --model TaCo-B

    # Export with a custom checkpoint and output folder
    python deploy/export_onnx.py --model TaCo-T --weights /path/to/ckpt.pth --out /tmp/taco_t
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import SearchExportWrapper, TACO_CONFIGS, TemplateExportWrapper, build_taco, load_taco_weights
from verify_utils import decode_center_head, file_sha256, hann2d, make_ort_session, rel_l2


def main() -> None:
    import onnx

    parser = argparse.ArgumentParser(description="Export TaCo PyTorch model to template.onnx and search.onnx.")
    parser.add_argument("--model", type=str, required=True, choices=list(TACO_CONFIGS.keys()))
    parser.add_argument(
        "--weights",
        "--checkpoint",
        dest="weights",
        type=Path,
        default=None,
        help="Path to PyTorch weight file (default: weights/<model>.pth).",
    )
    parser.add_argument("--random-init", action="store_true", help="Export random-initialized structure without loading weights.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: onnx_models/<model>).")
    parser.add_argument("--samples", type=int, default=4, help="Number of validation pairs to check against ORT FP32.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument("--no-reparam", action="store_true", help="Skip RepViT backbone reparameterization.")
    args = parser.parse_args()

    out_dir = args.out or (REPO_ROOT / "onnx_models" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    weights_path = None
    if not args.random_init:
        weights_path = args.weights or (REPO_ROOT / "weights" / f"{args.model}.pth")
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weight file not found: {weights_path}. Pass --weights <path> or --random-init."
            )

    model = build_taco(args.model, deploy=False)
    if weights_path is not None:
        load_taco_weights(model, weights_path)
    model = model.float().cpu().eval().requires_grad_(False)

    cfg = TACO_CONFIGS[args.model]
    zs, xs, feat = cfg["template_size"], cfg["search_size"], cfg["search_size"] // cfg["stride"]
    window = hann2d(feat, centered=True)

    # Deterministic validation inputs
    gen = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(args.samples):
        z = torch.randn(1, 3, zs, zs, generator=gen)
        x = torch.randn(1, 3, xs, xs, generator=gen)
        tbb = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
        samples.append((z, x, tbb))

    with torch.no_grad():
        full_refs = []
        for z, x, tbb in samples:
            out = model(z, x, tbb)
            full_refs.append((out["score_map"].numpy(), out["size_map"].numpy(), out["offset_map"].numpy()))

        reparam_stats = model.switch_to_deploy(reparam_backbone=(not args.no_reparam and cfg["reparam_body"]))

        z0, x0, tbb0 = samples[0]
        cache0 = model.backbone.flatten_cache(model.forward_template(z0, tbb0))
        cache_names = [name for name, _ in cache0]
        cache_tensors0 = tuple(t for _, t in cache0)

        tpl_wrapper = TemplateExportWrapper(model).eval()
        srch_wrapper = SearchExportWrapper(model).eval()

        tpl_path = out_dir / "template.onnx"
        srch_path = out_dir / "search.onnx"

        import inspect
        export_kwargs = dict(opset_version=args.opset, do_constant_folding=True)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        torch.onnx.export(
            tpl_wrapper,
            (z0, tbb0),
            str(tpl_path),
            input_names=["z", "tbb"],
            output_names=cache_names,
            **export_kwargs,
        )
        torch.onnx.export(
            srch_wrapper,
            (x0, *cache_tensors0),
            str(srch_path),
            input_names=["search"] + cache_names,
            output_names=["score", "size", "offset"],
            **export_kwargs,
        )
        onnx.checker.check_model(str(tpl_path))
        onnx.checker.check_model(str(srch_path))

    # Numerical verification: ORT FP32 chain vs original PyTorch model
    tpl_sess = make_ort_session(tpl_path, threads=2)
    srch_sess = make_ort_session(srch_path, threads=2)
    checks = []
    for i, ((z, x, tbb), (ref_score, ref_size, ref_offset)) in enumerate(zip(samples, full_refs)):
        ort_caches = tpl_sess.run(cache_names, {"z": z.numpy(), "tbb": tbb.numpy()})
        ort_score, ort_size, ort_offset = srch_sess.run(
            ["score", "size", "offset"],
            {"search": x.numpy(), **dict(zip(cache_names, ort_caches))},
        )
        ort_prob = np.clip(1.0 / (1.0 + np.exp(-ort_score.astype(np.float64))), 1e-4, 1.0 - 1e-4)
        e_score = rel_l2(ort_prob, ref_score)
        e_size = rel_l2(np.clip(ort_size, 1e-4, 1.0 - 1e-4), ref_size)
        e_offset = rel_l2(ort_offset, ref_offset)

        ref_cell, ref_box = decode_center_head(
            np.log(ref_score / (1.0 - ref_score)), ref_size, ref_offset, window
        )
        ort_cell, ort_box = decode_center_head(ort_score, ort_size, ort_offset, window)
        box_err = float(np.max(np.abs(ort_box - ref_box)))
        row = {
            "sample": i,
            "prob_rel_l2": e_score,
            "size_rel_l2": e_size,
            "offset_rel_l2": e_offset,
            "peak_match": ort_cell == ref_cell,
            "box_max_abs": box_err,
        }
        checks.append(row)
        assert max(e_score, e_size, e_offset) <= 1e-4, f"ORT mismatch on sample {i}: {row}"
        assert ort_cell == ref_cell and box_err <= 1e-5, f"Decoded box mismatch on sample {i}: {row}"

    io_meta = {
        "model": args.model,
        "weights_sha256": file_sha256(weights_path) if weights_path is not None else None,
        "template_size": zs,
        "search_size": xs,
        "template_factor": cfg["template_factor"],
        "search_factor": cfg["search_factor"],
        "stride": cfg["stride"],
        "feat_size": feat,
        "box_norm_legacy": False,
        "cache_names": cache_names,
        "cache_shapes": {n: list(t.shape) for n, t in cache0},
        "search_outputs": {"score": [1, 1, feat, feat], "size": [1, 2, feat, feat], "offset": [1, 2, feat, feat]},
        "reparam_body": reparam_stats,
        "template_onnx_sha256": file_sha256(tpl_path),
        "search_onnx_sha256": file_sha256(srch_path),
        "ort_verification": checks,
    }
    (out_dir / "io.json").write_text(json.dumps(io_meta, indent=2) + "\n")
    print(f"Exported {args.model} to {out_dir}")
    print(f"  template.onnx ({tpl_path.stat().st_size / 1e6:.2f} MB) sha256={io_meta['template_onnx_sha256'][:12]}...")
    print(f"  search.onnx   ({srch_path.stat().st_size / 1e6:.2f} MB) sha256={io_meta['search_onnx_sha256'][:12]}...")
    print(f"  ORT verification passed on {len(checks)}/{len(checks)} samples (max rel L2 <= 1e-4).")


if __name__ == "__main__":
    main()
