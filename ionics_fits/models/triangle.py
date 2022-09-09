import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

from .. import Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


class Triangle(Model):
    """Triangle function according to:
    y(x>=x0) = k_p*|x-x0| + y0
    y(x<x0) = k_m*|x-x0| + y0
    y = max(y, y_min)
    y = min(y, m_max)
    k = 0.5*(k_p + k_m)
    dk = 0.5*(k_p - k_m)

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis y0
      - k: average slope
      - dk: slope difference (asymmetry parameter, fixed to 0 by default)
      - y_min: minimum value of y (bound to -inf by default)
      - y_max: maximum value of y (bound to +inf by default)

    Derived parameters:
      - k_m: slope for x < x0
      - k_p: slope for x >= x0
    """

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        k: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale / x_scale),
        dk: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale / x_scale),
        y_min: ModelParameter(
            fixed_to=-np.inf, scale_func=lambda x_scale, y_scale, _: y_scale
        ),
        y_max: ModelParameter(
            fixed_to=+np.inf, scale_func=lambda x_scale, y_scale, _: y_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        k_p = k + dk
        k_m = k - dk

        y_p = k_p * np.abs(x - x0) + y0
        y_m = k_m * np.abs(x - x0) + y0
        y = np.where(x >= x0, y_p, y_m)
        y = np.where(y > y_min, y, y_min)
        y = np.where(y < y_max, y, y_max)

        return y

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Sets initial values for model parameters based on heuristics. Typically
        called during `Fitter.fit`.

        Heuristic results should stored in :param model_parameters: using the
        `ModelParameter`'s `initialise` method. This ensures that all information passed
        in by the user (fixed values, initial values, bounds) is used correctly.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        # Written to be handle the case of data which is only well-modelled by a
        # triangle function near `x0` but saturates further away
        model_parameters["y_max"].initialise(max(y))
        model_parameters["y_min"].initialise(min(y))

        min_ind = np.argmin(y)
        max_ind = np.argmax(y)

        y_min = y[min_ind]
        y_max = y[max_ind]
        x_min = x[min_ind]
        x_max = x[max_ind]

        # Case 1: positive slope with peaks left and right of x_min above y_min
        left_peak_ind = np.argmax(y[x <= x_min])
        right_peak_ind = np.argmax(y[x >= x_min])
        k_m_p = (y[left_peak_ind] - y_min) / (x_min - x[left_peak_ind])
        k_p_p = (y[right_peak_ind] - y_min) / (x[right_peak_ind] - x_min)

        positive_prominence = max(y[left_peak_ind], y[right_peak_ind]) - y_min

        # Case 2: negative slope with peaks left and right of x_max below y_max
        left_peak_ind = np.argmin(y[x <= x_max])
        right_peak_ind = np.argmin(y[x >= x_max])
        k_m_n = (y_max - y[left_peak_ind]) / (x_max - x[left_peak_ind])
        k_p_n = (y_max - y[right_peak_ind]) / (x[right_peak_ind] - x_max)

        negative_prominence = y_max - max(y[left_peak_ind], y[right_peak_ind])

        if positive_prominence > negative_prominence:
            model_parameters["x0"].initialise(x_min)
            model_parameters["y0"].initialise(y_min)
            model_parameters["k"].initialise(0.5 * (k_p_p + k_m_p))
            model_parameters["dk"].initialise(0.5 * (k_p_p - k_m_p))
        else:
            model_parameters["x0"].initialise(x_min)
            model_parameters["y0"].initialise(y_min)
            model_parameters["k"].initialise(0.5 * (k_p_n + k_m_n))
            model_parameters["dk"].initialise(0.5 * (k_p_n - k_m_n))

    @staticmethod
    def calculate_derived_params(
        fitted_params: Dict[str, float], fit_uncertainties: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        derived_params = {}
        derived_params["k_p"] = fitted_params["k"] + fitted_params["dk"]
        derived_params["k_m"] = fitted_params["k"] - fitted_params["dk"]

        derived_uncertainties = {}
        derived_uncertainties["kp"] = np.sqrt(
            fit_uncertainties["k"] ** 2 + fit_uncertainties["k"] ** 2
        )
        derived_uncertainties["km"] = derived_uncertainties["kp"]

        return derived_params, derived_uncertainties
