from __future__ import annotations

import numpy as np

"""
Calibration helper functions.

What this file does:
1. Provides logit/sigmoid transforms.
2. Adjusts probabilities to match target prevalence assumptions.

Possible improvements:
1. Add isotonic and Platt calibration options for comparison.
2. Validate calibration quality with reliability diagrams.
"""


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
