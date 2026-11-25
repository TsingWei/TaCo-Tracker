"""Bounding-box geometry for the training crops."""

import torch


def jitter_box(box, scale_jitter, center_jitter):
    """Apply log-normal size noise and uniform center noise to an xywh box."""
    size = box[2:] * torch.exp(torch.randn(2) * scale_jitter)
    max_offset = size.prod().sqrt() * center_jitter
    center = box[:2] + box[2:] / 2 + max_offset * (torch.rand(2) - 0.5)
    return torch.cat((center - size / 2, size))


def flip_box(box, width=1):
    """Mirror an xywh box across a pixel or normalized coordinate extent."""
    return torch.stack((width - box[0] - box[2], box[1], box[2], box[3]))
