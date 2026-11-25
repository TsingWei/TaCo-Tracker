"""Exact deploy-time reparameterization for RepViT backbones.

Fuses ``Conv2d_BN`` and multi-branch ``RepVGGDW`` modules into plain biased
``nn.Conv2d`` layers without changing the mathematical function (up to FP32
rounding), and remaps ``_StageView`` references accordingly.
"""

from typing import Dict

import torch
import torch.nn as nn

from .repvit import Conv2d_BN, RepVGGDW, Residual


def _bump(stats: Dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _fuse(mod: nn.Module, stats: Dict[str, int], remap: dict) -> None:
    for name, child in list(mod.named_children()):
        if isinstance(child, (Conv2d_BN, RepVGGDW, Residual)) and callable(getattr(child, "fuse", None)):
            fused = child.fuse()
            if fused is not child:
                setattr(mod, name, fused)
                remap[id(child)] = (child, fused)
                _bump(stats, "fuse_" + type(child).__name__)
            _fuse(fused, stats, remap)
        else:
            _fuse(child, stats, remap)


def _views(body: nn.Module):
    for m in body.modules():
        for val in list(vars(m).values()):
            for item in (val if isinstance(val, (list, tuple)) else [val]):
                if hasattr(item, "op_list"):
                    yield item


def _remap_views(body: nn.Module, remap: dict, stats: Dict[str, int]) -> None:
    for view in _views(body):
        new_ops = [remap[id(op)][1] if id(op) in remap else op for op in view.op_list]
        stats["view_ops_remapped"] = stats.get("view_ops_remapped", 0) + sum(
            a is not b for a, b in zip(new_ops, view.op_list)
        )
        view.op_list = new_ops
    registered = {id(m) for m in body.modules()}
    stale = [
        type(op).__name__
        for view in _views(body)
        for op in view.op_list
        if isinstance(op, nn.Module) and id(op) not in registered
    ]
    if stale:
        raise RuntimeError(f"Stage views still reference unregistered modules: {stale}")


@torch.no_grad()
def reparam_body(body: nn.Module) -> Dict[str, int]:
    """Fuse all RepViT Conv2d_BN / RepVGGDW modules inside ``body`` in place."""
    stats: Dict[str, int] = {}
    remap: dict = {}
    _fuse(body, stats, remap)
    _remap_views(body, remap, stats)
    left = sorted(
        {
            type(m).__name__
            for m in body.modules()
            if isinstance(m, (Conv2d_BN, RepVGGDW))
        }
    )
    if left:
        raise RuntimeError(f"Unfused RepViT modules remaining in body: {left}")
    stats["standalone_bn2d_left"] = sum(isinstance(m, nn.BatchNorm2d) for m in body.modules())
    return stats
