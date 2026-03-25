from __future__ import annotations

import numpy as np

"""Calibration helpers used to shift probabilities to a target prevalence."""


def logit(p: np.ndarray) -> np.ndarray:
    # Clip probabilities to avoid log(0) numerical errors.
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prevalence_shift_calibration(
    probabilities: np.ndarray,
    train_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """Adjust probabilities by applying an intercept shift for target population prevalence."""
    # Intercept shift in log-odds space.
    delta = np.log(target_prevalence / (1 - target_prevalence)) - np.log(train_prevalence / (1 - train_prevalence))
    return sigmoid(logit(probabilities) + delta)
