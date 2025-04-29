from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..normal import NormalFitter
from ..utils import scale_no_rescale
from . import heuristics
from .triangle import Triangle


class ConeSlice(Model):
    """Slice through a cone according to::

        z = sign * sqrt( (k_x * (x - x0))**2 + (k_y * (y - y0)) ** 2)) + z0

    This model represents a slice through the cone with fixed ``y``, given by::

        z = sign(k_x) * sqrt( (k_x * (x - x0))**2 + alpha ** 2 ) + z0

    where:

      * ``alpha = k_y * (y - y0)``
      * we use the sign of ``k_x`` to set the sign for the cone

    Floating ``z0`` and ``alpha`` without a user-estimate for either may result in an
    unreliable fit.

    See :meth:`_func` for parameter details.
    """

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [False], [False]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        x0: ModelParameter(scale_func=scale_no_rescale),
        z0: ModelParameter(scale_func=scale_no_rescale, fixed_to=0),
        k: ModelParameter(scale_func=scale_no_rescale),
        alpha: ModelParameter(lower_bound=0, scale_func=scale_no_rescale),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param z0: vertical offset to the cone
        :param k: slope along ``x``
        :param alpha: offset due to being off-centre in the y-axis
        """
        return np.sign(k) * np.sqrt((k * (x - x0)) ** 2 + alpha**2) + z0

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        fit = NormalFitter(x=x, y=y, model=Triangle())
        self.parameters["x0"].heuristic = fit.values["x0"]
        self.parameters["k"].heuristic = k = fit.values["k"]

        peak_value = fit.values["y0"]  # peak_value = alpha * k + z0

        if self.parameters["z0"].has_user_initial_value():
            z0 = self.parameters["z0"].get_initial_value()
            self.parameters["alpha"].heuristic = (peak_value - z0) / k
        elif self.parameters["alpha"].has_user_initial_value():
            alpha = self.parameters["alpha"].get_initial_value()
            self.parameters["z0"].heuristic = peak_value - alpha * k
        else:  # Hope one of z0 / alpha dominates the other
            _, z0_cost = heuristics.param_min_sqrs(
                model=self,
                x=x,
                y=y,
                scanned_param="z0",
                scanned_param_values=[peak_value],
                defaults={"alpha": 0},
            )
            _, alpha_cost = heuristics.param_min_sqrs(
                model=self,
                x=x,
                y=y,
                scanned_param="alpha",
                scanned_param_values=[peak_value / k],
                defaults={"z0": 0},
            )
            if z0_cost < alpha_cost:
                self.parameters["z0"].heuristic = peak_value
                self.parameters["alpha"].heuristic = 0
            else:
                self.parameters["z0"].heuristic = 0
                self.parameters["alpha"].heuristic = peak_value
