"""RYS — empirical CKA connectome and Repeat-Your-Self surgery on Llama-3.2-3B.

Public re-exports for convenience inside the notebook.
"""

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix, cka_matrix_bootstrap
from rys.data import csqa_prompts, gsm8k_prompts, wikitext_prompts
from rys.modules import change_points, leiden_communities, plateau_metric
from rys.plots import connectome_heatmap, delta_heatmap, panel
from rys.surgery import apply_rys

__all__ = [
    "apply_rys",
    "capture_residual_stream",
    "change_points",
    "cka_matrix",
    "cka_matrix_bootstrap",
    "connectome_heatmap",
    "csqa_prompts",
    "delta_heatmap",
    "gsm8k_prompts",
    "leiden_communities",
    "panel",
    "plateau_metric",
    "wikitext_prompts",
]
