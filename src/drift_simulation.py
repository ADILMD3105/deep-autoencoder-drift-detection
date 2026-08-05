import numpy as np


def gradual_drift(data, start_ratio=0.0, drift_strength=0.5):
    """
    Gradually shifts feature means over time.
    """
    drifted_data = data.copy()
    n_samples, n_features = drifted_data.shape

    start_index = int(n_samples * start_ratio)

    for i in range(start_index, n_samples):
        factor = (i - start_index) / (n_samples - start_index)
        drifted_data[i] += factor * drift_strength

    return drifted_data


def sudden_drift(data, drift_strength=1.0):
    """
    Applies a sudden shift to all samples.
    """
    return data + drift_strength


def feature_specific_drift(data, feature_idx, drift_strength=1.0):
    """
    Drifts a single feature.
    """
    drifted_data = data.copy()
    drifted_data[:, feature_idx] += drift_strength
    return drifted_data
