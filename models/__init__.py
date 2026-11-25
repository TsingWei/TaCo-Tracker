"""TaCo (Target-aligned Correlation) PyTorch model architecture definitions.

Usage:
    from models import build_taco, TACO_CONFIGS

    model = build_taco("TaCo-T")  # or "TaCo-S", "TaCo-B"
    model.eval()

    # Strictly initialize only the shared CNN body before training.
    model.backbone.load_pretrained_backbone("pretrained_models/lowformer_E1.pt")
"""

from .head import RepConvBN, RepConvCenterHead
from .taco import (
    TACO_CONFIGS,
    FiLM,
    SearchExportWrapper,
    TaCoBackbone,
    TaCoTrackerModel,
    TargetAlignedCorrelation,
    TemplateExportWrapper,
    box_parts,
    box_pool,
    build_taco,
    load_taco_weights,
    patch_coverage_mask,
)

__all__ = [
    "TACO_CONFIGS",
    "FiLM",
    "RepConvBN",
    "RepConvCenterHead",
    "SearchExportWrapper",
    "TaCoBackbone",
    "TaCoTrackerModel",
    "TargetAlignedCorrelation",
    "TemplateExportWrapper",
    "box_parts",
    "box_pool",
    "build_taco",
    "load_taco_weights",
    "patch_coverage_mask",
]
