"""Explainability: IG, LRP stub, probes, sanity checks, faithfulness, perturbations."""
from .integrated_gradients import ig_attribution
from .faithfulness import deletion_curve, insertion_curve
from .perturbations import band_limited_attenuation, suppress_lag_range, remove_channel_group
from .probes import linear_probe_accuracy, probe_all_layers
from .sanity_checks import randomize_parameters, cascading_randomization

__all__ = [
    "ig_attribution",
    "deletion_curve", "insertion_curve",
    "band_limited_attenuation", "suppress_lag_range", "remove_channel_group",
    "linear_probe_accuracy", "probe_all_layers",
    "randomize_parameters", "cascading_randomization",
]
