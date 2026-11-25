"""LowFormer-E1 backbone for TaCo-T.

Adapted from the official LowFormer implementation (Nottebaum et al., "Beyond MACs:
Exploiting Hardware-Aware Efficiency in Hybrid Vision Models", Apache-2.0 License).
Only the first three stages (strides 4, 8, 16) are used in TaCo-T; the stride-32
stage is omitted by default.
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _val2list(x, repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def _val2tuple(x, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = _val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def _build_norm(name: Optional[str] = "bn2d", num_features: Optional[int] = None) -> Optional[nn.Module]:
    if name == "bn2d":
        return nn.BatchNorm2d(num_features)
    if name == "ln":
        return nn.LayerNorm(num_features)
    return None


def _build_act(name: Optional[str]) -> Optional[nn.Module]:
    act_map = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": partial(nn.GELU, approximate="tanh"),
    }
    if name in act_map:
        return act_map[name]()
    return None


def _get_same_padding(kernel_size: int) -> int:
    assert kernel_size % 2 > 0, "kernel_size must be odd"
    return kernel_size // 2


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act: Optional[str] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = _build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            return x
        if self.shortcut is None:
            return self.forward_main(x)
        res = self.forward_main(x) + self.shortcut(x)
        if self.post_act is not None:
            res = self.post_act(res)
        return res


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
    ):
        super().__init__()
        padding = _get_same_padding(kernel_size) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = _build_norm(norm, num_features=out_channels)
        self.act = _build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        grouping: int = 1,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super().__init__()
        use_bias = _val2tuple(use_bias, 3)
        norm = _val2tuple(norm, 3)
        act_func = _val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels, mid_channels, 1, stride=1, groups=grouping,
            norm=norm[0], act_func=act_func[0], use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels, mid_channels, kernel_size, stride=stride, groups=mid_channels,
            norm=norm[1], act_func=act_func[1], use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1, groups=grouping,
            norm=norm[2], act_func=act_func[2], use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        groups: int = 1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = _val2tuple(use_bias, 2)
        norm = _val2tuple(norm, 2)
        act_func = _val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels, mid_channels, kernel_size, stride=stride, groups=groups,
            use_bias=use_bias[0], norm=norm[0], act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1,
            use_bias=use_bias[1], norm=norm[1], act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.spatial_conv(x))


class LowFormerConvBlock(nn.Module):
    """Purely convolutional LowFormer-E1 stage-3 block (attention and MLP removed)."""

    def __init__(
        self,
        in_channels: int,
        expand_ratio: float = 4,
        norm: str = "bn2d",
        act_func: str = "hswish",
        fuseconv: bool = True,
        grouping: int = 1,
    ):
        super().__init__()
        context_module = nn.Identity()
        if fuseconv and in_channels < 256:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                norm=norm,
                act_func=(act_func, None),
            )
        else:
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                grouping=grouping,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            )
        self.total = nn.Sequential(context_module, ResidualBlock(local_module, IdentityLayer()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)


class LowFormerBackbone(nn.Module):
    """LowFormer-E1 backbone exposing ``input_stem`` and ``stages`` (strides 4, 8, 16)."""

    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels: int = 3,
        expand_ratio: float = 4,
        norm: str = "bn2d",
        act_func: str = "hswish",
        fastit: bool = True,
        bigit: bool = True,
        grouping: int = 1,
        drop_s4: bool = True,
    ):
        super().__init__()
        # Stage 0: stride-2 stem
        stem_ops = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self._build_local_block(
                width_list[0], width_list[0], stride=1, expand_ratio=2,
                norm=norm, act_func=act_func, fusedmbconv=fastit, grouping=grouping,
            )
            stem_ops.append(ResidualBlock(block, IdentityLayer()))
        self.input_stem = OpSequential(stem_ops)
        in_ch = width_list[0]

        stages: List[nn.Module] = []
        # Stages 1 and 2 (strides 4 and 8)
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            ops = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self._build_local_block(
                    in_ch, w, stride=stride,
                    expand_ratio=6 if (stride == 2 and bigit) else expand_ratio,
                    norm=norm, act_func=act_func, fusedmbconv=fastit, grouping=grouping,
                )
                ops.append(ResidualBlock(block, IdentityLayer() if stride == 1 else None))
                in_ch = w
            stages.append(OpSequential(ops))

        # Stage 3 (stride 16) and optionally Stage 4 (stride 32)
        tail_widths = width_list[3:4] if drop_s4 else width_list[3:]
        tail_depths = depth_list[3:4] if drop_s4 else depth_list[3:]
        for w, d in zip(tail_widths, tail_depths):
            ops = []
            down = self._build_local_block(
                in_ch, w, stride=2, expand_ratio=6 if bigit else expand_ratio,
                norm=norm, act_func=act_func, fusedmbconv=fastit, grouping=grouping,
            )
            ops.append(ResidualBlock(down, None))
            in_ch = w
            for _ in range(d):
                ops.append(
                    LowFormerConvBlock(
                        in_channels=in_ch,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                        fuseconv=fastit,
                        grouping=grouping,
                    )
                )
            stages.append(OpSequential(ops))

        self.stages = nn.ModuleList(stages)
        self.widths = list(width_list)

    @staticmethod
    def _build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fusedmbconv: bool = True,
        grouping: int = 1,
    ) -> nn.Module:
        if fusedmbconv:
            return FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=False,
                norm=norm,
                act_func=(act_func, None),
            )
        return MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            grouping=grouping,
            use_bias=False,
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )


def build_lowformer_e1(drop_s4: bool = True) -> Tuple[LowFormerBackbone, List[int]]:
    """Build LowFormer-E1 (widths [20, 40, 80, 160, 320], depths [0, 1, 1, 4, 4])."""
    width_list = [20, 40, 80, 160, 320]
    depth_list = [0, 1, 1, 4, 4]
    backbone = LowFormerBackbone(
        width_list=width_list,
        depth_list=depth_list,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        fastit=True,
        bigit=True,
        grouping=1,
        drop_s4=drop_s4,
    )
    return backbone, width_list
