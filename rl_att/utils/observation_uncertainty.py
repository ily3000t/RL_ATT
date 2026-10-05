"""Inverse observation boxes for |y-x| <= rho*|x| + beta.

This is a numerical threat-model helper, not a posterior over SUMO states.
Sampling uses a caller-owned RNG and does not enforce physical feature domains.
"""

import numpy as np


def inverse_observation_box(observation, relative_scale=.2, absolute_scale=.05):
    """Return the exact coordinatewise bounds for 0 <= rho < 1, beta >= 0."""
    if (not np.isscalar(relative_scale) or not np.isscalar(absolute_scale) or
            not np.isfinite(relative_scale) or not np.isfinite(absolute_scale) or
            not 0 <= relative_scale < 1 or absolute_scale < 0):
        raise ValueError("Inverse box requires finite 0 <= rho < 1 and beta >= 0")
    observed = np.asarray(observation, dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("Observation must be finite")
    rho, beta = relative_scale, absolute_scale
    lower = np.where(observed >= beta, (observed - beta) / (1 + rho), (observed - beta) / (1 - rho))
    upper = np.where(observed <= -beta, (observed + beta) / (1 + rho), (observed + beta) / (1 - rho))
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Inverse bounds overflowed")
    return lower, upper


def sample_inverse_observations(observation, samples, rng, relative_scale=.2, absolute_scale=.05):
    """Uniform box candidates, shape (..., samples, features); not belief weights."""
    observed = np.asarray(observation, dtype=np.float64)
    if observed.ndim < 1 or observed.shape[-1] == 0 or type(samples) is not int or samples <= 0:
        raise ValueError("Provide feature vectors and a positive integer sample count")
    if rng is None or not hasattr(rng, "uniform"):
        raise ValueError("Provide an explicit local NumPy RNG")
    lower, upper = inverse_observation_box(observed, relative_scale, absolute_scale)
    unit = rng.uniform(0., 1., size=observed.shape[:-1] + (samples, observed.shape[-1]))
    return lower[..., None, :] + unit * (upper - lower)[..., None, :]
