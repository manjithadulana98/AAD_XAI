"""Explainability: IG, GradCAM, SHAP, LIME, probes, sanity checks, faithfulness, perturbations."""
from .integrated_gradients import ig_attribution
from .faithfulness import deletion_curve, insertion_curve
from .perturbations import band_limited_attenuation, suppress_lag_range, remove_channel_group
from .probes import linear_probe_accuracy, probe_all_layers
from .sanity_checks import randomize_parameters, cascading_randomization
from .gradcam import gradcam_attribution, gradcam_all_blocks, gradcam_temporal_heatmap
from .shap_explainer import shap_deep_attribution, shap_kernel_attribution
from .lime_explainer import lime_attribution, lime_batch_attribution
from .probes_vlaai import (
    extract_all_activations_pt,
    attention_decoding_probes,
    auditory_feature_probes,
    linguistic_feature_probes,
    run_all_probes_pt,
)
from .probe_viz import (
    plot_probe_accuracy_by_layer,
    plot_probe_comparison,
    plot_auditory_probes,
    save_probe_results,
)

__all__ = [
    "ig_attribution",
    "deletion_curve", "insertion_curve",
    "band_limited_attenuation", "suppress_lag_range", "remove_channel_group",
    "linear_probe_accuracy", "probe_all_layers",
    "randomize_parameters", "cascading_randomization",
    # GradCAM
    "gradcam_attribution", "gradcam_all_blocks", "gradcam_temporal_heatmap",
    # SHAP
    "shap_deep_attribution", "shap_kernel_attribution",
    # LIME
    "lime_attribution", "lime_batch_attribution",
    # VLAAI probes
    "extract_all_activations_pt",
    "attention_decoding_probes", "auditory_feature_probes",
    "linguistic_feature_probes", "run_all_probes_pt",
    # Visualization
    "plot_probe_accuracy_by_layer", "plot_probe_comparison",
    "plot_auditory_probes", "save_probe_results",
]
