"""Shared 128-channel RepConv center head for TaCo.

During training, each 3x3 layer is a multi-branch ``RepConvBN`` block
(3x3+BN, 1x1+BN, and optional identity+BN). Calling ``switch_to_deploy()``
algebraically folds all branches into a single biased 3x3 ``nn.Conv2d``.
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_score_prior(conv: nn.Conv2d, prior: float) -> None:
    """Focal-loss prior initialization for the score logit bias."""
    if prior and prior > 0:
        nn.init.constant_(conv.bias, -math.log((1.0 - prior) / prior))


class RepConvBN(nn.Module):
    """RepVGG-style block: ReLU(BN(3x3 x) + BN(1x1 x) [+ BN(x)])."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv3 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.conv1 = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn_id = nn.BatchNorm2d(cin) if cin == cout else None
        self.act = nn.ReLU(inplace=True)
        self.fused = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.act(self.fused(x))
        y = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    @staticmethod
    def _fold(kernel: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return kernel * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.fused is not None:
            return
        k3, b3 = self._fold(self.conv3.weight, self.bn3)
        k1, b1 = self._fold(F.pad(self.conv1.weight, [1, 1, 1, 1]), self.bn1)
        k, b = k3 + k1, b3 + b1
        if self.bn_id is not None:
            cin = self.conv3.in_channels
            eye = torch.zeros_like(self.conv3.weight)
            eye[torch.arange(cin), torch.arange(cin), 1, 1] = 1.0
            ki, bi = self._fold(eye, self.bn_id)
            k, b = k + ki, b + bi
        self.fused = nn.Conv2d(
            self.conv3.in_channels,
            self.conv3.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        ).to(self.conv3.weight.device)
        self.fused.weight.copy_(k)
        self.fused.bias.copy_(b)
        for name in ("conv3", "bn3", "conv1", "bn1", "bn_id"):
            setattr(self, name, None)


class RepConvCenterHead(nn.Module):
    """Center head with widths C -> c -> c/2 -> c/4 -> c/8 -> 1x1 output.

    In TaCo, ``channel=128`` and ``shared=2``: the first two RepConv blocks are
    shared across the score, offset, and size branches, while the remaining two
    RepConv blocks and 1x1 output projection are branch-specific.
    """

    def __init__(
        self,
        inplanes: int = 224,
        channel: int = 128,
        stride: int = 16,
        shared: int = 2,
        score_prior: float = 0.1,
    ):
        super().__init__()
        self.stride = int(stride)
        self.shared = int(shared)
        if not 0 <= self.shared <= 4:
            raise ValueError(f"shared must be in [0, 4], got {shared!r}")

        widths = [inplanes, channel, channel // 2, channel // 4, channel // 8]
        self.trunk = nn.Sequential(*[RepConvBN(widths[i], widths[i + 1]) for i in range(self.shared)])

        def _branch(n_out: int):
            layers = [RepConvBN(widths[i], widths[i + 1]) for i in range(self.shared, 4)]
            return nn.Sequential(*layers), nn.Conv2d(widths[4], n_out, kernel_size=1)

        self.tower_ctr, self.out_ctr = _branch(1)
        self.tower_offset, self.out_offset = _branch(2)
        self.tower_size, self.out_size = _branch(2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        set_score_prior(self.out_ctr, score_prior)

    def switch_to_deploy(self) -> None:
        """Fold all RepConvBN blocks into single 3x3 Conv2d layers."""
        for m in self.modules():
            if isinstance(m, RepConvBN):
                m.switch_to_deploy()

    def forward_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deployment forward returning raw score logits, sigmoid size map, and offset map.

        Keeping the score sigmoid on the host avoids FP16 saturation ties on edge NPUs.
        """
        x = self.trunk(x)
        score_logits = self.out_ctr(self.tower_ctr(x))
        size_map = self.out_size(self.tower_size(x)).sigmoid()
        offset_map = self.out_offset(self.tower_offset(x))
        return score_logits, size_map, offset_map

    def get_score_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clamped score probability map, clamped size map, and offset map."""
        x = self.trunk(x)
        score_ctr = self.out_ctr(self.tower_ctr(x))
        offset_map = self.out_offset(self.tower_offset(x))
        size_map = self.out_size(self.tower_size(x))
        return (
            torch.clamp(score_ctr.sigmoid(), min=1e-4, max=1.0 - 1e-4),
            torch.clamp(size_map.sigmoid(), min=1e-4, max=1.0 - 1e-4),
            offset_map,
        )

    def cal_bbox(
        self,
        score_map_ctr: torch.Tensor,
        feat_size: int,
        size_map: torch.Tensor,
        offset_map: torch.Tensor,
        return_score: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode the argmax cell into normalized (cx, cy, w, h) coordinates."""
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // feat_size
        idx_x = idx % feat_size

        idx_2 = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx_2).squeeze(-1)
        offset = offset_map.flatten(2).gather(dim=2, index=idx_2).squeeze(-1)

        bbox = torch.cat(
            [
                (idx_x.to(torch.float32) + offset[:, :1]) / feat_size,
                (idx_y.to(torch.float32) + offset[:, 1:]) / feat_size,
                size,
            ],
            dim=1,
        )
        if return_score:
            return bbox, max_score
        return bbox

    def forward(self, x: torch.Tensor, feat_size: int = 16):
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        bbox = self.cal_bbox(score_map_ctr, feat_size, size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map
