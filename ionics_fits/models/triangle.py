from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_invariant, scale_power, scale_x, scale_y


class Triangle(Model):
    """Triangle function according to::

        y(x>=x0) = k_p*|x-x0| + y0
        y(x<x0) = k_m*|x-x0| + y0
        y = max(y, y_min)
        y = min(y, m_max)
        k_p = (1 + sym) * k
        k_m = (1 - sym) * k

    See :meth:`_func` for parameter details.
    """

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [True]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        x0: ModelParameter(scale_func=scale_x()),
        y0: ModelParameter(scale_func=scale_y()),
        k: ModelParameter(scale_func=scale_power(x_power=-1, y_power=1)),
        sym: ModelParameter(
            lower_bound=-1, upper_bound=1, fixed_to=0, scale_func=scale_invariant
        ),
        y_min: ModelParameter(fixed_to=-np.inf, scale_func=scale_y()),
        y_max: ModelParameter(fixed_to=+np.inf, scale_func=scale_y()),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param y0: y-axis offset
        :param k: average slope
        :param sym: symmetry parameter (fixed to ``0`` by default)
        :param y_min: minimum value of y (bound to ``-inf`` by default)
        :param y_max: maximum value of y (bound to ``+inf`` by default)
        """
        k_p = k * (1 + sym)
        k_m = k * (1 - sym)

        y_p = k_p * np.abs(x - x0) + y0
        y_m = k_m * np.abs(x - x0) + y0
        y = np.where(x >= x0, y_p, y_m)
        y = np.where(y > y_min, y, y_min)
        y = np.where(y < y_max, y, y_max)

        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        # Written to handle the case of data which is only well-modelled by a
        # triangle function near `x0` but saturates further away
        self.parameters["y_max"].heuristic = max(y)
        self.parameters["y_min"].heuristic = min(y)

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

        positive_defaults = {
            "x0": x_min,
            "y0": y_min,
            "k": 0.5 * (k_p + k_m),
            "sym": (alpha - 1) / (1 + alpha),
        }
        positive_parameters = {
            param: param_data.get_initial_value(default=positive_defaults.get(param))
            for param, param_data in self.parameters.items()
        }
        positive_cost = np.sum((y - self._func(x, **positive_parameters)) ** 2)

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

        negative_defaults = {
            "x0": x_max,
            "y0": y_max,
            "k": 0.5 * (k_p + k_m),
            "sym": (alpha - 1) / (1 + alpha),
        }
        negative_parameters = {
            param: param_data.get_initial_value(default=negative_defaults.get(param))
            for param, param_data in self.parameters.items()
        }
        negative_cost = np.sum((y - self._func(x, **negative_parameters)) ** 2)

        if positive_cost < negative_cost:
            best_params = positive_parameters
        else:
            best_params = negative_parameters

        for param, value in best_params.items():
            self.parameters[param].heuristic = value

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Derived parameters:

        * k_m: slope for ``x < x0``
        * k_p: slope for ``x >= x0``
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
