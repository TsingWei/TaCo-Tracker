#!/usr/bin/env python3
"""Standalone PyTorch-free tracking metric evaluator (AUC, P_Norm, P).

Usage:
    python metrics.py TaCo-T --dataset lasot
    python metrics.py TaCo-B --dataset vasttrack_test --data-root /path/to/VastTrack/test
    python metrics.py --all
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DATASET_PATHS, MODELS, RESULTS_PATH
from datasets import get_dataset

NESTED_DATASETS = {
    "trackingnet",
    "got10k",
    "lasot",
    "lasot_extension_subset",
    "uav",
    "nfs",
    "vasttrack_test",
    "tnl2k",
}

DEFAULT_DATASETS = [
    "lasot",
    "vasttrack_test",
    "got10k_val",
    "lasot_extension_subset",
    "tnl2k",
    "uav",
    "uavdt",
    "dtb70",
    "nfs_30",
    "visdrone2018",
    "uavdark135",
]


def calc_err_center(pred_bb: np.ndarray, anno_bb: np.ndarray, normalized: bool = False) -> np.ndarray:
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)
    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]
    return np.sqrt(((pred_center - anno_center) ** 2).sum(axis=1))


def calc_iou_overlap(pred_bb: np.ndarray, anno_bb: np.ndarray) -> np.ndarray:
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = np.clip(br - tl + 1.0, a_min=0.0, a_max=None)
    intersection = sz[:, 0] * sz[:, 1]
    union = pred_bb[:, 2] * pred_bb[:, 3] + anno_bb[:, 2] * anno_bb[:, 3] - intersection
    return intersection / np.maximum(union, 1e-12)


def calc_seq_err_robust(
    pred_bb: np.ndarray,
    anno_bb: np.ndarray,
    dataset: str,
    target_visible: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_bb = pred_bb.astype(np.float64).copy()
    anno_bb = anno_bb.astype(np.float64)

    if np.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise ValueError("Invalid predictions (NaN or negative width/height).")
    if np.isnan(anno_bb).any() and dataset != "uav":
        raise ValueError("NaNs found in ground-truth annotation.")

    zero_wh = (pred_bb[:, 2:] == 0.0).any(axis=1)
    for i in range(1, pred_bb.shape[0]):
        if zero_wh[i] and not np.isnan(anno_bb[i, :]).any():
            pred_bb[i, :] = pred_bb[i - 1, :]

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if pred_bb.shape[0] > anno_bb.shape[0]:
            pred_bb = pred_bb[: anno_bb.shape[0], :]
        else:
            pad = np.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4), dtype=pred_bb.dtype)
            pred_bb = np.concatenate((pred_bb, pad), axis=0)

    pred_bb[0, :] = anno_bb[0, :]

    valid = (anno_bb[:, 2:] > 0.0).sum(axis=1) == 2
    if target_visible is not None:
        target_visible = target_visible.astype(bool)
        valid = np.logical_and(valid, target_visible)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_norm = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    err_center[~valid] = -1.0 if dataset == "uav" else np.inf
    err_center_norm[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == "lasot" and target_visible is not None:
        err_center[~target_visible] = np.inf
        err_center_norm[~target_visible] = np.inf

    if np.isnan(err_overlap).any():
        raise ValueError("NaNs in calculated IoU overlap.")
    return err_overlap, err_center, err_center_norm, valid


def load_result(result_file: str) -> np.ndarray:
    for delimiter in ("\t", ",", " "):
        try:
            data = np.loadtxt(result_file, delimiter=delimiter, dtype=np.float64)
            return np.atleast_2d(data)
        except Exception:
            continue
    raise ValueError(f"Could not parse prediction file: {result_file}")


def result_path(results_root: str, seq) -> str:
    if seq.dataset in NESTED_DATASETS:
        nested = os.path.join(results_root, seq.dataset, f"{seq.name}.txt")
        if os.path.isfile(nested):
            return nested
    return os.path.join(results_root, f"{seq.name}.txt")


def evaluate_dataset(
    model_name: str,
    dataset_name: str,
    dataset_paths: Optional[dict] = None,
    results_root: Optional[str] = None,
    sequence_filter: Optional[str] = None,
) -> Dict:
    paths = dict(DATASET_PATHS if dataset_paths is None else dataset_paths)
    dataset = get_dataset(dataset_name, paths)
    if sequence_filter is not None:
        dataset = [s for s in dataset if s.name == sequence_filter]
        if not dataset:
            raise ValueError(f"Sequence {sequence_filter!r} not found in {dataset_name!r}")

    root = results_root or os.path.join(RESULTS_PATH, model_name, dataset_name)
    th_overlap = np.arange(0.0, 1.0 + 0.05, 0.05, dtype=np.float64)
    th_center = np.arange(0, 51, dtype=np.float64)
    th_center_norm = np.arange(0, 51, dtype=np.float64) / 100.0

    overlap_curves: List[np.ndarray] = []
    center_curves: List[np.ndarray] = []
    center_norm_curves: List[np.ndarray] = []
    avg_overlaps: List[float] = []

    for seq in dataset:
        pred_file = result_path(root, seq)
        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"Missing prediction file: {pred_file}")

        pred_bb = load_result(pred_file)
        anno_bb = np.asarray(seq.ground_truth_rect, dtype=np.float64).reshape(-1, 4)
        visible = None if seq.target_visible is None else np.asarray(seq.target_visible)
        err_ov, err_c, err_cn, valid = calc_seq_err_robust(pred_bb, anno_bb, seq.dataset, visible)

        n = anno_bb.shape[0]
        overlap_curves.append((err_ov[:, None] > th_overlap[None, :]).sum(axis=0).astype(np.float64) / n)
        center_curves.append((err_c[:, None] <= th_center[None, :]).sum(axis=0).astype(np.float64) / n)
        center_norm_curves.append((err_cn[:, None] <= th_center_norm[None, :]).sum(axis=0).astype(np.float64) / n)
        avg_overlaps.append(float(err_ov[valid].mean()) if valid.any() else 0.0)

    ov_mat = np.stack(overlap_curves, axis=0)
    c_mat = np.stack(center_curves, axis=0)
    cn_mat = np.stack(center_norm_curves, axis=0)
    return {
        "model": model_name,
        "dataset": dataset_name,
        "num_sequences": len(dataset),
        "auc": float(ov_mat.mean(axis=0).mean()),
        "p_norm": float(cn_mat.mean(axis=0)[20]),
        "precision": float(c_mat.mean(axis=0)[20]),
        "avg_overlap": float(np.mean(avg_overlaps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AUC, P_Norm, and P for TaCo predictions (no PyTorch).")
    parser.add_argument("model_name", nargs="?", default=None, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to score.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset path from config.py.")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory.")
    parser.add_argument("--sequence", type=str, default=None, help="Score only a single sequence.")
    parser.add_argument("--all", action="store_true", help="Score all available models and datasets in results/.")
    args = parser.parse_args()

    if args.all:
        for model_name in MODELS:
            print(f"=== {model_name} ===")
            for ds in DEFAULT_DATASETS:
                rdir = os.path.join(RESULTS_PATH, model_name, ds)
                if not os.path.isdir(rdir):
                    continue
                try:
                    s = evaluate_dataset(model_name, ds)
                    print(
                        f"  {ds:24s} ({s['num_sequences']:4d} seqs): "
                        f"AUC={s['auc']*100:.2f}  P_Norm={s['p_norm']*100:.2f}  P={s['precision']*100:.2f}"
                    )
                except Exception as exc:
                    print(f"  {ds:24s}: ERROR ({exc})")
        return

    if not args.model_name or not args.dataset:
        parser.error("Specify <model_name> and --dataset, or pass --all.")

    paths = dict(DATASET_PATHS)
    if args.data_root:
        key_map = {
            "lasot": "lasot_path",
            "lasot_extension_subset": "lasot_extension_subset_path",
            "got10k_val": "got10k_path",
            "vasttrack_test": "vasttrack_test_path",
            "uav": "uav_path",
            "uavdt": "uavdt_path",
            "dtb70": "dtb70_path",
            "nfs": "nfs_path",
            "nfs_30": "nfs_path",
            "tnl2k": "tnl2k_path",
            "visdrone2018": "visdrone2018_path",
            "uavdark135": "uavdark135_path",
        }
        if args.dataset.lower() in key_map:
            paths[key_map[args.dataset.lower()]] = args.data_root

    s = evaluate_dataset(
        args.model_name,
        args.dataset,
        dataset_paths=paths,
        results_root=args.results_dir,
        sequence_filter=args.sequence,
    )
    print(
        f"{s['model']} on {s['dataset']} ({s['num_sequences']} seqs):\n"
        f"  AUC    : {s['auc'] * 100:.2f}%\n"
        f"  P_Norm : {s['p_norm'] * 100:.2f}%\n"
        f"  P      : {s['precision'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
