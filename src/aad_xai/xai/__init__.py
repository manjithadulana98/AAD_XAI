"""Explainability: IG, GradCAM, SHAP, LIME, probes, sanity checks, faithfulness, perturbations.

Submodules with heavy optional dependencies (captum, shap, lime, torch-based
probes) are imported defensively so that lightweight consumers -- e.g.
`composite_stability`, which only needs numpy/pandas/scipy -- can `import
aad_xai.xai` without requiring the full ML stack to be installed. A missing
optional dependency simply omits that submodule's names from this namespace
rather than failing the whole package import.
"""
import warnings

__all__ = []


def _try_import(module_name, names):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=names)
        for name in names:
            globals()[name] = getattr(module, name)
        __all__.extend(names)
    except ImportError as exc:
        warnings.warn(f"aad_xai.xai.{module_name} unavailable ({exc}); skipping {names}.", stacklevel=2)


_try_import("integrated_gradients", ["ig_attribution"])
_try_import("faithfulness", ["deletion_curve", "insertion_curve"])
_try_import("perturbations", ["band_limited_attenuation", "suppress_lag_range", "remove_channel_group"])
_try_import("probes", ["linear_probe_accuracy", "probe_all_layers"])
_try_import("sanity_checks", ["randomize_parameters", "cascading_randomization"])
_try_import("gradcam", ["gradcam_attribution", "gradcam_all_blocks", "gradcam_temporal_heatmap"])
_try_import("shap_explainer", ["shap_deep_attribution", "shap_kernel_attribution"])
_try_import("lime_explainer", ["lime_attribution", "lime_batch_attribution"])
_try_import("probes_vlaai", [
    "extract_all_activations_pt",
    "attention_decoding_probes",
    "auditory_feature_probes",
    "linguistic_feature_probes",
    "run_all_probes_pt",
])
_try_import("probe_viz", [
    "plot_probe_accuracy_by_layer",
    "plot_probe_comparison",
    "plot_auditory_probes",
    "save_probe_results",
])
_try_import("composite_stability", [
    "cohens_d",
    "bootstrap_ci",
    "wilcoxon_p",
    "fdr_correction",
    "load_montage_rois",
    "region_composite_test",
    "run_region_wise",
    "run_top_channels",
    "select_best_roi",
    "select_top_k_channels",
    "cross_validate_selection",
])

del _try_import
