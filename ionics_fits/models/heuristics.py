from typing import TYPE_CHECKING
import numpy as np
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


def get_sym_x(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_y_channels", "num_samples"), np.float64],
) -> float:
    """Returns `x_0` such that y(x-x_0) is maximally symmetric."""
    x_span = x.ptp()
    num_samples = x.size
    window_min = min(x) + 0.125 * x_span
    window_max = max(x) - 0.125 * x_span
    window = np.argwhere(np.logical_and(x >= window_min, x <= window_max)).squeeze()
    costs = np.zeros_like(window, dtype=float)

    for point_num, idx in np.ndenumerate(window.squeeze()):
        samples_left = idx
        samples_right = num_samples - idx - 1
        num_samples_idx = min(samples_left, samples_right)
        idx_left = idx - num_samples_idx
        idx_right = idx + num_samples_idx

        samples_left = y[:, idx_left:idx]
        samples_right = y[:, (idx + 1) : (idx_right + 1)]

        assert samples_left.shape == samples_right.shape

        diff = samples_left - np.fliplr(samples_right)

        # give more weight to windows with more structure
        mu = np.mean(samples_left)
        var = np.sum(np.square(samples_left - mu)) / samples_left.size

        cost = np.sqrt(np.sum(np.square(diff))) / samples_left.size
        costs[point_num] = cost / var

    return x[window[np.argmin(costs)]]
