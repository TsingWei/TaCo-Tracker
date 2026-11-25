"""RepViT-M1.1 and RepViT-M1.5 backbones for TaCo-S and TaCo-B.

Adapted from the official RepViT implementation (Wang et al., "RepViT: Revisiting
Mobile CNN From ViT Perspective", CVPR 2024). Only the first three stages (strides
4, 8, 16) are executed in TaCo; the stride-32 stage is omitted by default.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import SqueezeExcite
except ImportError:
    try:
        from timm.models.layers import SqueezeExcite
    except ImportError:
        class SqueezeExcite(nn.Module):
            """Pure-PyTorch drop-in matching timm.layers.SqueezeExcite defaults."""

            def __init__(self, in_chs: int, rd_ratio: float = 0.25):
                super().__init__()
                rd_chs = _make_divisible(in_chs * rd_ratio, 8)
                self.conv_reduce = nn.Conv2d(in_chs, rd_chs, 1, bias=True)
                self.act1 = nn.ReLU(inplace=True)
                self.conv_expand = nn.Conv2d(rd_chs, in_chs, 1, bias=True)
                self.gate = nn.Sigmoid()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_se = x.mean((2, 3), keepdim=True)
                x_se = self.conv_reduce(x_se)
                x_se = self.act1(x_se)
                x_se = self.conv_expand(x_se)
                return x * self.gate(x_se)


def _make_divisible(v: float, divisor: int, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ):
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x) * mask
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = F.pad(torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1), [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        if isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = F.pad(torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1), [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        return self


class RepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv = self.conv.fuse()
        conv1_w = F.pad(self.conv1.weight, [1, 1, 1, 1])
        identity = F.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
            [1, 1, 1, 1],
        )
        conv.weight.data.copy_(conv.weight + conv1_w + identity)
        conv.bias.data.copy_(conv.bias + self.conv1.bias)

        bn = self.bn
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(conv.weight * scale[:, None, None, None])
        conv.bias.data.copy_(bn.bias + (conv.bias - bn.running_mean) * scale)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp: int, hidden_dim: int, oup: int, kernel_size: int, stride: int, use_se: int, use_hs: int):
        super().__init__()
        assert stride in (1, 2)
        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU(),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU(),
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


REPVIT_CFGS = {
    # RepViT-M1.1 (used in TaCo-S): 13 blocks in stage 3 after the stride-2 downsample
    "m1_1": [
        # k, t, c, SE, HS, s
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ],
    # RepViT-M1.5 (used in TaCo-B): 25 blocks in stage 3 after the stride-2 downsample
    "m1_5": [
        # k, t, c, SE, HS, s
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ],
}


class _StageView:
    """Non-Module view over consecutive blocks so ``state_dict`` keeps ``features.N.*`` names."""

    def __init__(self, ops: List[nn.Module]):
        self.op_list = list(ops)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class RepViTBody(nn.Module):
    """RepViT feature extractor split into ``input_stem`` and ``stages`` (s1, s2, s3[, s4])."""

    def __init__(self, cfgs: list, drop_s4: bool = True):
        super().__init__()
        down = [i for i, row in enumerate(cfgs, start=1) if row[5] == 2]
        if len(down) != 3:
            raise ValueError(f"RepViTBody expects 3 stride-2 downsamples, got {len(down)}")

        active_cfgs = cfgs[: down[2] - 1] if drop_s4 else cfgs
        input_channel = active_cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers: List[nn.Module] = [patch_embed]
        for k, t, c, use_se, use_hs, s in active_cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(RepViTBlock(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        feats = list(self.features)
        self.input_stem = _StageView(feats[:1])
        cuts = ([1] + down[:2] + [len(feats)]) if drop_s4 else ([1] + down + [len(feats)])
        self.stages = [_StageView(feats[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
        full_cuts = [1] + down + [len(cfgs) + 1]
        self.widths = [cfgs[0][2]] + [_make_divisible(cfgs[b - 2][2], 8) for b in full_cuts[1:]]


def build_repvit(variant: str = "m1_1", drop_s4: bool = True) -> Tuple[RepViTBody, List[int]]:
    if variant not in REPVIT_CFGS:
        raise ValueError(f"Unknown RepViT variant {variant!r}; choose from {sorted(REPVIT_CFGS)}")
    body = RepViTBody(REPVIT_CFGS[variant], drop_s4=drop_s4)
    return body, body.widths
