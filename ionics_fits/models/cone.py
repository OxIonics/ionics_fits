from typing import Tuple, TYPE_CHECKING
import numpy as np

from . import heuristics, Triangle
from .. import common, Model, ModelParameter, NormalFitter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class ConeSlice(Model):
    """Slice through a cone.

    We parametrise cones as:
        z = sign * sqrt( (k_x * (x - x0))**2 + (k_y * (y - y0)) ** 2)) + z0

    This model represents a slice through the cone with fixed `y`, given by:
        z = sign(k_x) * sqrt( (k_x * (x - x0))**2 + alpha ** 2 ) + z0
    where:
      - alpha = k_y * (y - y0)
      - we use the sign of k_x to set the sign for the cone

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - z0: vertical offset to the cone. Fixed to 0 by default
      - alpha: offset due to being off-centre in the y-axis
      - k: slope along x

    Floating z0 and alpha without a user-estimate for either may result in an unreliable
    fit.
    """

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[bool, bool]:
        return False, False

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=common.scale_x),
        z0: ModelParameter(scale_func=common.scale_y, fixed_to=0),
        k: ModelParameter(scale_func=common.scale_y),
        alpha: ModelParameter(
            lower_bound=0, scale_func=common.scale_power(x_power=1, y_power=1)
        ),
    ) -> Array[("num_samples",), np.float64]:
        return np.sign(k) * np.sqrt((k * (x - x0)) ** 2 + alpha**2) + z0

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        y = y.squeeze()

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
