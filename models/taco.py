"""TaCo: Target-aligned Correlation architecture definition.

Implements the complete TaCo model family (TaCo-T, TaCo-S, TaCo-B):
  1. Shared three-stage CNN backbone (strides 4, 8, 16; stride-32 omitted):
       - TaCo-T: LowFormer-E1  (4 stage-3 modulation sites)
       - TaCo-S: RepViT-M1.1   (13 stage-3 modulation sites)
       - TaCo-B: RepViT-M1.5   (25 stage-3 modulation sites)
  2. Target-guided modulation (FiLM before every stage-3 block on the search stream).
  3. Target-aligned correlation (4x4 RoIAlign foreground kernels + 4x4 average-pooled
     context kernels -> 32 kernels, BatchNorm2d, Lite Kernel Weighting MLP, 1x1 -> 64 ch).
  4. Shared 128-channel RepConv center head (first 2 RepConv layers shared).

Tracking decomposes into a once-per-video template stage and a per-frame search stage:
    theta_z = g(z, b),   (score, size, offset) = f(x; theta_z).
"""

from pathlib import Path
from pickle import UnpicklingError
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align as _tv_roi_align
except ImportError:
    _tv_roi_align = None

from .backbones import build_lowformer_e1, build_repvit, reparam_body
from .head import RepConvCenterHead

TACO_CONFIGS = {
    "TaCo-T": {
        "body": "lowformer",
        "variant": "E1",
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
        "cond_dim": 256,
        "pwc_parts": 4,
        "pwc_ctx": 4,
        "pwc_adj": 64,
        "head_channels": 128,
        "head_shared": 2,
        "reparam_body": False,
        "modulation_sites": 4,
    },
    "TaCo-S": {
        "body": "repvit",
        "variant": "m1_1",
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
        "cond_dim": 256,
        "pwc_parts": 4,
        "pwc_ctx": 4,
        "pwc_adj": 64,
        "head_channels": 128,
        "head_shared": 2,
        "reparam_body": True,
        "modulation_sites": 13,
    },
    "TaCo-B": {
        "body": "repvit",
        "variant": "m1_5",
        "template_size": 128,
        "search_size": 256,
        "template_factor": 2.0,
        "search_factor": 4.0,
        "stride": 16,
        "cond_dim": 256,
        "pwc_parts": 4,
        "pwc_ctx": 4,
        "pwc_adj": 64,
        "head_channels": 128,
        "head_shared": 2,
        "reparam_body": True,
        "modulation_sites": 25,
    },
}


_REPVIT_SE_NAMES = {"conv_reduce": "fc1", "conv_expand": "fc2"}


def _rename_repvit_se(state, prefix, names):
    """Keep RepViT squeeze-excitation keys consistent with the released weights."""
    for key in list(state):
        if key.startswith(prefix + "features.") and ".token_mixer.1." in key:
            parent, layer, field = key.rsplit(".", 2)
            if layer in names:
                state.setdefault(f"{parent}.{names[layer]}.{field}", state[key])
                del state[key]


def _save_fallback_repvit_state(module, state, prefix, local_metadata):
    _rename_repvit_se(state, prefix, _REPVIT_SE_NAMES)


def _load_fallback_repvit_state(module, state, prefix, *args):
    _rename_repvit_se(state, prefix, {value: key for key, value in _REPVIT_SE_NAMES.items()})


# ---------------------------------------------------------------------------
# Template box pooling and kernel sampling helpers
# ---------------------------------------------------------------------------
def patch_coverage_mask(template_bb: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    """Fractional per-cell coverage inside ``template_bb`` -> ``(B, (H/p)*(W/p))``.

    ``template_bb`` has shape ``(B, 4)`` in normalized ``xyxy`` crop coordinates.
    """
    if height % patch_size or width % patch_size:
        raise ValueError(f"Template size {height}x{width} is not divisible by patch_size {patch_size}")
    scale = torch.tensor([width, height, width, height], device=template_bb.device, dtype=template_bb.dtype)
    bb = (template_bb * scale).clamp(min=0)
    x0, y0, x1, y1 = bb.unbind(-1)
    xs = torch.arange(width, device=template_bb.device, dtype=template_bb.dtype)
    ys = torch.arange(height, device=template_bb.device, dtype=template_bb.dtype)
    x_mask = ((xs[None, :] >= x0[:, None]) & (xs[None, :] < x1[:, None])).to(bb.dtype)
    y_mask = ((ys[None, :] >= y0[:, None]) & (ys[None, :] < y1[:, None])).to(bb.dtype)
    mask = y_mask[:, :, None] * x_mask[:, None, :]  # (B, H, W)
    p = patch_size
    return mask.unfold(1, p, p).unfold(2, p, p).mean(dim=(3, 4)).flatten(1)


def box_pool(feat: torch.Tensor, template_bb: torch.Tensor, img_size: int) -> torch.Tensor:
    """Coverage-weighted mean of template feature ``feat`` inside ``template_bb`` -> ``(B, C)``."""
    _, _, h, w = feat.shape
    cov = patch_coverage_mask(template_bb.float(), img_size, img_size, img_size // h)
    denom = cov.sum(dim=1, keepdim=True)
    weights = torch.where(denom > 1e-6, cov / denom.clamp_min(1e-6), torch.full_like(cov, 1.0 / (h * w)))
    return torch.einsum("bn,bcn->bc", weights.to(feat.dtype), feat.flatten(2))


def _roi_align_pure_pytorch(
    feat: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    output_size: int,
    spatial_scale: float,
    sampling_ratio: int = 2,
) -> torch.Tensor:
    """Exact pure-PyTorch fallback for aligned=True RoIAlign when torchvision is unavailable."""
    b, c, h, w = feat.shape
    k = int(output_size)
    sr = int(sampling_ratio)
    out = feat.new_zeros((b, c, k, k), dtype=torch.float32)
    feat_f = feat.float()
    xs_grid = torch.arange(w, device=feat.device, dtype=torch.float32)
    ys_grid = torch.arange(h, device=feat.device, dtype=torch.float32)
    steps = torch.arange(k * sr, device=feat.device, dtype=torch.float32) + 0.5

    for i in range(b):
        x0, y0, x1, y1 = boxes_xyxy[i] * spatial_scale - 0.5
        sx = (x0 + steps * (x1 - x0) / float(k * sr)).clamp(0.0, float(w - 1))
        sy = (y0 + steps * (y1 - y0) / float(k * sr)).clamp(0.0, float(h - 1))
        wx = (1.0 - (sx[:, None] - xs_grid[None, :]).abs()).clamp(min=0.0).view(k, sr, w).mean(dim=1)  # (k, w)
        wy = (1.0 - (sy[:, None] - ys_grid[None, :]).abs()).clamp(min=0.0).view(k, sr, h).mean(dim=1)  # (k, h)
        # (c, h, w) -> (c, k, k)
        out[i] = torch.einsum("chw,yh,xw->cyx", feat_f[i], wy, wx)
    return out


def box_parts(feat: torch.Tensor, template_bb: torch.Tensor, img_size: int, grid: int) -> torch.Tensor:
    """``grid x grid`` bilinear RoIAlign inside ``template_bb`` -> ``(B, grid*grid, C)``.

    Uses ``torchvision.ops.roi_align`` by default (which also exports cleanly to the
    standard ONNX ``RoiAlign`` operator). If ``torchvision`` lacks a compiled kernel
    for the active device (e.g., CPU-only torchvision with CUDA torch or MPS), falls
    back cleanly to CPU ``roi_align``; if ``torchvision`` is not installed, uses an
    exact pure-PyTorch bilinear fallback.
    """
    b = feat.shape[0]
    boxes = template_bb.float().clamp(0, 1) * img_size
    spatial_scale = feat.shape[-1] / float(img_size)

    if _tv_roi_align is not None:
        batch_idx = torch.arange(b, device=feat.device, dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_idx, boxes], dim=1)
        try:
            out = _tv_roi_align(
                feat.float(),
                rois,
                output_size=(grid, grid),
                spatial_scale=spatial_scale,
                sampling_ratio=2,
                aligned=True,
            )
        except (NotImplementedError, RuntimeError):
            # Common when torchvision C++/CUDA extension does not match PyTorch device
            out = _tv_roi_align(
                feat.float().cpu(),
                rois.cpu(),
                output_size=(grid, grid),
                spatial_scale=spatial_scale,
                sampling_ratio=2,
                aligned=True,
            ).to(feat.device)
    else:
        out = _roi_align_pure_pytorch(feat, boxes, output_size=grid, spatial_scale=spatial_scale, sampling_ratio=2)

    return out.to(feat.dtype).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Target-guided modulation and Target-aligned correlation
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Channel-wise affine modulation ``x * (1 + gamma) + beta`` (zero-initialized)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def params(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict cached ``(scale, shift)`` of shape ``(B, C, 1, 1)`` each."""
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1.0 + gamma)[..., None, None], beta[..., None, None]

    @staticmethod
    def modulate(x: torch.Tensor, sb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        scale, shift = sb
        return x * scale.to(x.dtype) + shift.to(x.dtype)


class TargetAlignedCorrelation(nn.Module):
    """Target-aligned correlation with Lite Kernel Weighting.

    Template side (once per video):
      - ``parts x parts`` RoIAlign target kernels (16) + ``ctx x ctx`` avg-pooled
        context kernels (16) -> 32 kernels scaled by ``C^-0.5``.
      - Lite Kernel Weighting MLP maps normalized kernel features concatenated with
        fractional in-box coverage ``a_i`` (1.0 for target kernels) to per-kernel
        scalar gain ``w_i = 1 + g_i`` and bias ``b_i``.

    Search side (per frame):
      - Single MatMul ``R = K X`` -> ``BatchNorm2d`` -> ``w * R + b`` -> 1x1 conv to ``out_ch``.
    """

    def __init__(self, channels: int, out_ch: int = 64, parts: int = 4, ctx: int = 4):
        super().__init__()
        self.parts = int(parts)
        self.ctx = int(ctx)
        self.n_kernels = self.parts ** 2 + self.ctx ** 2
        self.scale = channels ** -0.5

        self.bn_in = nn.BatchNorm2d(self.n_kernels)
        self.kf_norm = nn.LayerNorm(channels)
        self.kf_mlp = nn.Sequential(
            nn.Linear(channels + 1, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.kf_mlp[-1].weight)
        nn.init.zeros_(self.kf_mlp[-1].bias)
        self.adjust = nn.Conv2d(self.n_kernels, out_ch, kernel_size=1)

    def template_side(
        self,
        fz: torch.Tensor,
        template_bb: torch.Tensor,
        img_size: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute cached kernels ``(B, 32, C)`` and weights ``((B, 32, 1, 1), (B, 32, 1, 1))``."""
        target_tok = box_parts(fz, template_bb, img_size, self.parts)
        context_tok = F.adaptive_avg_pool2d(fz, self.ctx).flatten(2).transpose(1, 2)
        tok = torch.cat([target_tok, context_tok], dim=1)

        target_cov = torch.ones(fz.shape[0], self.parts ** 2, device=fz.device, dtype=torch.float32)
        context_cov = patch_coverage_mask(template_bb.float(), img_size, img_size, img_size // self.ctx)
        cov = torch.cat([target_cov, context_cov], dim=1)

        h = torch.cat([self.kf_norm(tok), cov.to(tok.dtype)[..., None]], dim=-1)
        g, b = self.kf_mlp(h).chunk(2, dim=-1)
        kernel_weights = ((1.0 + g)[..., None], b[..., None])
        return tok * self.scale, kernel_weights

    def forward(
        self,
        x: torch.Tensor,
        kernels: torch.Tensor,
        kernel_weights: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            r = torch.matmul(kernels.float(), x.float().flatten(2)).view(b, -1, h, w)
            r = self.bn_in(r)
        r = r.to(x.dtype)
        w_gain, w_bias = kernel_weights
        r = r * w_gain.to(r.dtype) + w_bias.to(r.dtype)
        return self.adjust(r)


# ---------------------------------------------------------------------------
# TaCo Backbone and Full Model
# ---------------------------------------------------------------------------
class TaCoBackbone(nn.Module):
    """Two-stream shared CNN backbone with stage-3 FiLM and target-aligned correlation."""

    def __init__(
        self,
        body: str = "lowformer",
        variant: str = "E1",
        template_size: int = 128,
        cond_dim: int = 256,
        pwc_parts: int = 4,
        pwc_ctx: int = 4,
        pwc_adj: int = 64,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.body_type = body
        self.variant = variant
        self.template_size = int(template_size)
        self.freeze_bn = freeze_bn

        if body == "lowformer":
            self.body, self.widths = build_lowformer_e1(drop_s4=True)
        elif body == "repvit":
            self.body, self.widths = build_repvit(variant=variant, drop_s4=True)
            if any(".token_mixer.1.conv_reduce." in key for key, _ in self.body.named_parameters()):
                # The pure-PyTorch fallback uses different names for equivalent layers.
                self.body.register_state_dict_post_hook(_save_fallback_repvit_state)
                self.body.register_load_state_dict_pre_hook(_load_fallback_repvit_state)
        else:
            raise ValueError(f"Unsupported TaCo backbone body: {body!r}")

        c_s3 = self.widths[3]
        self.out_dim = c_s3
        self.embed_dim = self.out_dim + int(pwc_adj)

        # Number of stage-3 blocks after the stride-2 downsample (4 / 13 / 25 for T / S / B)
        n_s3_blocks = len(self.body.stages[2].op_list) - 1
        self.num_modulation_sites = n_s3_blocks

        # Target-guided modulation (box-pooled descriptor -> shared MLP -> per-site FiLM)
        self.cond = nn.Sequential(
            nn.LayerNorm(c_s3),
            nn.Linear(c_s3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.film = nn.ModuleDict(
            {"s3": nn.ModuleDict({str(j): FiLM(cond_dim, c_s3) for j in range(n_s3_blocks)})}
        )

        # Projection and target-aligned correlation at stride 16
        self.final_conv = nn.Sequential(
            nn.Conv2d(c_s3, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
        )
        self.pwc_head = nn.ModuleDict(
            {"s3": TargetAlignedCorrelation(c_s3, out_ch=pwc_adj, parts=pwc_parts, ctx=pwc_ctx)}
        )
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn and mode:
            for module in self.body.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def load_pretrained_backbone(self, weights_path: Union[str, Path]) -> Tuple[int, int]:
        """Strictly initialize strides 4, 8 and 16, returning file and model tensor counts."""
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model"):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
        if not isinstance(state, dict):
            raise TypeError(f"Expected a backbone state dict in {weights_path}")
        expected = self.body.state_dict()
        if self.body_type == "lowformer":
            tensors = {key.removeprefix("backbone."): value for key, value in state.items()}
        else:
            tensors = {key: value for key, value in state.items() if key.startswith("features.")}
            _rename_repvit_se(tensors, "", _REPVIT_SE_NAMES)
        retained = {key: value for key, value in tensors.items() if key in expected}
        loaded = len(retained)
        if self.body_type == "repvit":
            # The official ImageNet releases leave out these tracking counters
            # for the RepVGGDW depthwise branch, while supplying its BN statistics.
            for key in expected.keys() - retained.keys():
                if key.endswith(".token_mixer.0.conv.bn.num_batches_tracked"):
                    retained[key] = torch.zeros_like(expected[key])
        # Reject every other missing key, including counters PyTorch may supply.
        missing = expected.keys() - retained.keys()
        if missing:
            raise RuntimeError(f"Missing backbone tensors in {weights_path}: {sorted(missing)}")
        self.body.load_state_dict(retained, strict=True)
        return loaded, len(expected)

    def _run_body(
        self,
        x: torch.Tensor,
        film_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run stages 1..3 (strides 4, 8, 16); apply FiLM before every stage-3 block if provided."""
        f = self.body.stages[0](self.body.input_stem(x))
        f = self.body.stages[1](f)
        stage3 = self.body.stages[2]
        f = stage3.op_list[0](f)  # stride-2 downsample to stride 16
        for j, blk in enumerate(stage3.op_list[1:]):
            if film_cache is not None and j in film_cache:
                f = FiLM.modulate(f, film_cache[j])
            f = blk(f)
        return f

    def extract_template_cache(self, z: torch.Tensor, template_bb: torch.Tensor) -> dict:
        """Once-per-video template stage: compute all cached operands ``theta_z``."""
        fz = self._run_body(z, film_cache=None)
        desc = box_pool(fz, template_bb, self.template_size)
        cond = self.cond(desc)

        film_s3: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for j, film_mod in self.film["s3"].items():
            film_s3[int(j)] = film_mod.params(cond)

        kernels, kernel_weights = self.pwc_head["s3"].template_side(fz, template_bb, self.template_size)
        return {
            "film_s3": film_s3,
            "pwc_s3": kernels,
            "pwcmod_s3": kernel_weights,
        }

    def forward_search(self, x: torch.Tensor, cache: dict) -> torch.Tensor:
        """Per-frame search stage: return concatenated modulated feature and correlation response."""
        fx = self._run_body(x, film_cache=cache["film_s3"])
        feat = self.final_conv(fx)
        corr = self.pwc_head["s3"](fx, cache["pwc_s3"], cache["pwcmod_s3"])
        return torch.cat([feat, corr], dim=1)

    def flatten_cache(self, cache: dict) -> List[Tuple[str, torch.Tensor]]:
        """Return cached operands in the exact canonical ONNX I/O order."""
        rows: List[Tuple[str, torch.Tensor]] = []
        # Match lexicographic key ordering used in exported ONNX graphs:
        # sorted(keys, key=str) over ('s3', j) tuples.
        ordered_idx = sorted(cache["film_s3"].keys(), key=lambda j: str(("s3", j)))
        for j in ordered_idx:
            scale, shift = cache["film_s3"][j]
            rows.append((f"film_s3_{j}_scale", scale))
            rows.append((f"film_s3_{j}_shift", shift))
        rows.append(("pwc_s3", cache["pwc_s3"]))
        k_scale, k_shift = cache["pwcmod_s3"]
        rows.append(("pwcmod_s3_scale", k_scale))
        rows.append(("pwcmod_s3_shift", k_shift))
        return rows

    def unflatten_cache(self, flat_tensors: Tuple[torch.Tensor, ...]) -> dict:
        """Reconstruct structured cache dict from canonical flat ONNX inputs."""
        ordered_idx = sorted(range(self.num_modulation_sites), key=lambda j: str(("s3", j)))
        expected = 2 * len(ordered_idx) + 3
        if len(flat_tensors) != expected:
            raise ValueError(f"Expected {expected} cache tensors, got {len(flat_tensors)}")
        film_s3: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        pos = 0
        for j in ordered_idx:
            film_s3[j] = (flat_tensors[pos], flat_tensors[pos + 1])
            pos += 2
        pwc_s3 = flat_tensors[pos]
        pwcmod_s3 = (flat_tensors[pos + 1], flat_tensors[pos + 2])
        return {"film_s3": film_s3, "pwc_s3": pwc_s3, "pwcmod_s3": pwcmod_s3}


class TaCoTrackerModel(nn.Module):
    """Complete TaCo tracking network combining ``TaCoBackbone`` and ``RepConvCenterHead``."""

    def __init__(self, variant_name: str = "TaCo-T", freeze_bn: bool = True):
        super().__init__()
        if variant_name not in TACO_CONFIGS:
            raise ValueError(f"Unknown variant {variant_name!r}; choose from {list(TACO_CONFIGS)}")
        self.variant_name = variant_name
        cfg = TACO_CONFIGS[variant_name]
        self.cfg = dict(cfg)

        self.backbone = TaCoBackbone(
            body=cfg["body"],
            variant=cfg["variant"],
            template_size=cfg["template_size"],
            cond_dim=cfg["cond_dim"],
            pwc_parts=cfg["pwc_parts"],
            pwc_ctx=cfg["pwc_ctx"],
            pwc_adj=cfg["pwc_adj"],
            freeze_bn=freeze_bn,
        )
        self.box_head = RepConvCenterHead(
            inplanes=self.backbone.embed_dim,
            channel=cfg["head_channels"],
            stride=cfg["stride"],
            shared=cfg["head_shared"],
            score_prior=0.1,
        )
        self.feat_size = cfg["search_size"] // cfg["stride"]

    def fresh_param_predicate(self, name: str) -> bool:
        """Identify parameters outside the ImageNet-initialized body."""
        return not name.startswith("backbone.body.")

    def switch_to_deploy(self, reparam_backbone: Optional[bool] = None) -> Dict[str, int]:
        """Reparameterize RepConv head blocks and (for RepViT variants) backbone blocks."""
        if reparam_backbone is None:
            reparam_backbone = bool(self.cfg["reparam_body"])
        stats: Dict[str, int] = {}
        if reparam_backbone and self.backbone.body_type == "repvit":
            stats = reparam_body(self.backbone.body)
        self.box_head.switch_to_deploy()
        return stats

    def forward_template(self, template: torch.Tensor, template_bb: torch.Tensor) -> dict:
        return self.backbone.extract_template_cache(template, template_bb)

    def forward_search(
        self,
        search: torch.Tensor,
        cache: dict,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feat = self.backbone.forward_search(search, cache)
        if return_logits:
            score_logits, size_map, offset_map = self.box_head.forward_logits(feat)
            return {
                "score_logits": score_logits,
                "size_map": size_map,
                "offset_map": offset_map,
            }
        score_map, pred_boxes, size_map, offset_map = self.box_head(feat, feat_size=self.feat_size)
        return {
            "score_map": score_map,
            "pred_boxes": pred_boxes,
            "size_map": size_map,
            "offset_map": offset_map,
        }

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        template_bb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cache = self.forward_template(template, template_bb)
        return self.forward_search(search, cache, return_logits=False)


class TemplateExportWrapper(nn.Module):
    """Wrapper matching the exported ``template.onnx`` signature: ``(z, tbb) -> (*caches,)``."""

    def __init__(self, model: TaCoTrackerModel):
        super().__init__()
        self.bb = model.backbone

    def forward(self, z: torch.Tensor, tbb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cache = self.bb.extract_template_cache(z, tbb)
        return tuple(t for _, t in self.bb.flatten_cache(cache))


class SearchExportWrapper(nn.Module):
    """Wrapper matching the exported ``search.onnx`` signature: ``(search, *caches) -> (score, size, offset)``."""

    def __init__(self, model: TaCoTrackerModel):
        super().__init__()
        self.bb = model.backbone
        self.head = model.box_head

    def forward(self, search: torch.Tensor, *flat_caches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self.bb.unflatten_cache(flat_caches)
        feat = self.bb.forward_search(search, cache)
        score, size, offset = self.head.forward_logits(feat)
        # Match the trained predictor's size bounds for saturated checkpoints.
        return score, size.clamp(1e-4, 1.0 - 1e-4), offset


def load_taco_weights(model: TaCoTrackerModel, weights_path: Union[str, Path]) -> TaCoTrackerModel:
    """Load a TaCo ``.pth`` weight file (FP16-stored or FP32) into ``model`` in FP32."""
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except UnpicklingError:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "net" in ckpt:
        state = ckpt["net"]
    else:
        state = ckpt
    if model.backbone.body_type == "repvit":
        state = state.copy()
        _rename_repvit_se(state, "backbone.body.", _REPVIT_SE_NAMES)

    model_keys = set(model.state_dict().keys())
    filtered = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in state.items()
        if k in model_keys
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading {weights_path}: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {weights_path}: {unexpected[:10]}")
    return model


def build_taco(
    variant: str = "TaCo-T",
    deploy: bool = False,
    weights: Optional[Union[str, Path]] = None,
    freeze_bn: bool = True,
) -> TaCoTrackerModel:
    """Instantiate a TaCo variant (``TaCo-T``, ``TaCo-S``, or ``TaCo-B``)."""
    model = TaCoTrackerModel(variant_name=variant, freeze_bn=freeze_bn)
    if weights is not None:
        load_taco_weights(model, weights)
    if deploy:
        model.eval()
        model.switch_to_deploy()
    return model
