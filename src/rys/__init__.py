"""RYS — empirical generated-response CKA connectomes and RYS surgery.

Public re-exports for convenience inside the notebook.
"""

from rys.activations import capture_generated_residual_stream, capture_residual_stream
from rys.cka import cka_matrix, cka_matrix_bootstrap
from rys.data import csqa_prompts, gsm8k_prompts, mmlu_prompts
from rys.modules import change_points, leiden_communities, plateau_metric
from rys.plots import connectome_heatmap, delta_heatmap, panel
from rys.residual_force import (
    amplification_long,
    predict_cka_under_rys,
    residual_force_long,
    residual_force_matrices,
)
from rys.surgery import apply_rys

__all__ = [
    "amplification_long",
    "apply_rys",
    "capture_generated_residual_stream",
    "capture_residual_stream",
    "change_points",
    "cka_matrix",
    "cka_matrix_bootstrap",
    "connectome_heatmap",
    "csqa_prompts",
    "delta_heatmap",
    "gsm8k_prompts",
    "leiden_communities",
    "mmlu_prompts",
    "panel",
    "plateau_metric",
    "predict_cka_under_rys",
    "residual_force_long",
    "residual_force_matrices",
]
