"""Generalized IoU, L1 and CenterNet focal supervision for TaCo."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xywh_to_xyxy(box):
    x, y, w, h = box.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def giou_loss(boxes1, boxes2):
    """Return per-box GIoU loss and IoU for corresponding xyxy boxes."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=-1)
    intersection = (torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
                    - torch.maximum(boxes1[:, :2], boxes2[:, :2])).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union
    enclosure = (torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
                 - torch.minimum(boxes1[:, :2], boxes2[:, :2])).clamp_min(0).prod(dim=-1)
    return 1 - (iou - (enclosure - union) / enclosure), iou


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive = target.eq(1).float()
        negative = target.lt(1).float()
        weights = (1 - target).pow(self.beta)
        prediction = prediction.clamp_min(1e-7)
        inverse = (1 - prediction).clamp_min(1e-7)
        positive_loss = prediction.log() * inverse.pow(self.alpha) * positive
        negative_loss = inverse.log() * prediction.pow(self.alpha) * weights * negative
        return -(positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)


def _gaussian_radius(size, overlap=0.7):
    width, height = size.unbind(-1)
    b1 = height + width
    c1 = width * height * (1 - overlap) / (1 + overlap)
    r1 = (b1 + (b1.square() - 4 * c1).sqrt()) / 2
    b2 = 2 * (height + width)
    c2 = (1 - overlap) * width * height
    r2 = (b2 + (b2.square() - 16 * c2).sqrt()) / 2
    a3 = 4 * overlap
    b3 = -2 * overlap * (height + width)
    c3 = (overlap - 1) * width * height
    r3 = (b3 + (b3.square() - 4 * a3 * c3).sqrt()) / 2
    return torch.minimum(r1, torch.minimum(r2, r3)).clamp_min(0).to(torch.int32)


def generate_heatmap(bboxes, heatmap_size=16):
    """Generate one (B, H, W) CenterNet Gaussian map per [N, B, 4] box set."""
    maps = []
    for boxes in bboxes:
        scaled = boxes.detach().cpu() * heatmap_size
        size = scaled[:, 2:]
        centers = (scaled[:, :2] + size / 2).round().to(torch.int32)
        heatmap = torch.zeros(len(boxes), heatmap_size, heatmap_size)
        for plane, (x, y), radius in zip(heatmap, centers.tolist(), _gaussian_radius(size).tolist()):
            left, right = min(x, radius), min(heatmap_size - x, radius + 1)
            top, bottom = min(y, radius), min(heatmap_size - y, radius + 1)
            if left + right <= 0 or top + bottom <= 0:
                continue
            dy, dx = np.ogrid[-top:bottom, -left:right]
            sigma = (2 * radius + 1) / 6
            gaussian = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            plane[y - top:y + bottom, x - left:x + right] = torch.as_tensor(gaussian)
        maps.append(heatmap.to(boxes.device))
    return maps


class TaCoLoss(nn.Module):
    def __init__(self, giou_weight=2.0, l1_weight=5.0, focal_weight=1.0):
        super().__init__()
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss()

    def forward(self, outputs, search_boxes):
        scores = outputs["score_map"].float()
        heatmap = generate_heatmap(search_boxes.unsqueeze(0), scores.shape[-1])[0]
        pred = box_cxcywh_to_xyxy(outputs["pred_boxes"].float().reshape(-1, 4))
        target = box_xywh_to_xyxy(search_boxes.float()).to(pred.device).clamp(0, 1)
        giou, iou = giou_loss(pred, target)
        giou = giou.mean()
        l1 = F.l1_loss(pred, target)
        focal = self.focal(scores, heatmap.to(scores.device).unsqueeze(1))
        total = self.giou_weight * giou + self.l1_weight * l1 + self.focal_weight * focal
        return total, {"Loss/total": total.detach(), "Loss/giou": giou.detach(),
                       "Loss/l1": l1.detach(), "Loss/focal": focal.detach(),
                       "IoU": iou.detach().mean()}
