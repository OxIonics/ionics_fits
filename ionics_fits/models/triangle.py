import copy
from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

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
    k_p = (1 + sym) * k
    k_m = (1 - sym) * k

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis offset
      - k: average slope
      - sym: symmetry parameter (fixed to 0 by default)
      - y_min: minimum value of y (bound to -inf by default)
      - y_max: maximum value of y (bound to +inf by default)

    Derived parameters:
      - k_m: slope for x < x0
      - k_p: slope for x >= x0
    """

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        k: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale / x_scale),
        sym: ModelParameter(lower_bound=-1, upper_bound=1, fixed_to=0),
        y_min: ModelParameter(
            fixed_to=-np.inf, scale_func=lambda x_scale, y_scale, _: y_scale
        ),
        y_max: ModelParameter(
            fixed_to=+np.inf, scale_func=lambda x_scale, y_scale, _: y_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        k_p = k * (1 + sym)
        k_m = k * (1 - sym)

        y_p = k_p * np.abs(x - x0) + y0
        y_m = k_m * np.abs(x - x0) + y0
        y = np.where(x >= x0, y_p, y_m)
        y = np.where(y > y_min, y, y_min)
        y = np.where(y < y_max, y, y_max)

        return y

    # pytype: enable=invalid-annotation
    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        # Written to handle the case of data which is only well-modelled by a
        # triangle function near `x0` but saturates further away
        model_parameters["y_max"].heuristic = max(y)
        model_parameters["y_min"].heuristic = min(y)

        min_ind = np.argmin(y)
        max_ind = np.argmax(y)

        y_min = y[min_ind]
        y_max = y[max_ind]
        x_min = x[min_ind]
        x_max = x[max_ind]

        # Case 1: positive slope with peaks left and right of x_min above y_min
        x_l = x[x <= x_min]
        x_r = x[x >= x_min]
        y_l = y[x <= x_min]
        y_r = y[x >= x_min]

        left_peak_ind = np.argmax(y_l)
        right_peak_ind = np.argmax(y_r)

        dx_l = x_min - x_l[left_peak_ind]
        dx_r = x_r[right_peak_ind] - x_min

        k_m = (y_l[left_peak_ind] - y_min) / dx_l if dx_l != 0 else 0
        k_p = (y_r[right_peak_ind] - y_min) / dx_r if dx_r != 0 else 0
        alpha = 0 if k_m == 0 else k_p / k_m

        positive_parameters = copy.deepcopy(model_parameters)
        positive_parameters["x0"].heuristic = x_min
        positive_parameters["y0"].heuristic = y_min

        positive_parameters["k"].heuristic = 0.5 * (k_p + k_m)
        positive_parameters["sym"].heuristic = (alpha - 1) / (1 + alpha)

        positive_parameters = {
            param: param_data.get_initial_value()
            for param, param_data in positive_parameters.items()
        }
        positive_residuals = np.sum(
            np.power(y - self._func(x, **positive_parameters), 2)
        )

        # Case 2: negative slope with peaks left and right of x_max below y_max
        x_l = x[x <= x_max]
        x_r = x[x >= x_max]
        y_l = y[x <= x_max]
        y_r = y[x >= x_max]

        left_peak_ind = np.argmin(y_l)
        right_peak_ind = np.argmin(y_r)

        dx_l = x_max - x_l[left_peak_ind]
        dx_r = x_r[right_peak_ind] - x_max

        k_m = (y_l[left_peak_ind] - y_max) / dx_l if dx_l != 0 else 0
        k_p = (y_r[right_peak_ind] - y_max) / dx_r if dx_r != 0 else 0
        alpha = 0 if k_m == 0 else k_p / k_m

        negative_parameters = copy.deepcopy(model_parameters)
        negative_parameters["x0"].heuristic = x_max
        negative_parameters["y0"].heuristic = y_max
        negative_parameters["k"].heuristic = 0.5 * (k_p + k_m)
        negative_parameters["sym"].heuristic = (alpha - 1) / (1 + alpha)

        negative_parameters = {
            param: param_data.get_initial_value()
            for param, param_data in negative_parameters.items()
        }
        negative_residuals = np.sum(
            np.power(y - self._func(x, **negative_parameters), 2)
        )

        if positive_residuals < negative_residuals:
            best_params = positive_parameters
        else:
            best_params = negative_parameters

        for param, value in best_params.items():
            model_parameters[param].heuristic = value

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param x: x-axis data
        :param y: y-axis data
        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        derived_params = {}
        k = fitted_params["k"]
        sym = fitted_params["sym"]

        k_err = fitted_params["k"]
        sym_err = fitted_params["sym"]

        derived_params["k_p"] = (1 + sym) * k
        derived_params["k_m"] = (1 - sym) * k

        derived_uncertainties = {}
        derived_uncertainties["k_p"] = np.sqrt(
            (k_err * (1 + sym)) ** 2 + (k * sym_err) ** 2
        )
        derived_uncertainties["k_m"] = np.sqrt(
            (k_err * (1 - sym)) ** 2 + (k * sym_err) ** 2
        )

        return derived_params, derived_uncertainties
