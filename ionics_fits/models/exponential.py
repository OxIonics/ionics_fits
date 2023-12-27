from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .. import common, Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


class Exponential(Model):
    """Exponential function according to:
    y(x < x_dead) = y0
    y(x >= x_dead) = y0 + (y_inf-y0)*(1-exp(-(x-x_dead)/tau))

    Fit parameters (all floated by default unless stated otherwise):
      - x_dead: x-axis "dead time" (fixed to 0 by default)
      - y0: initial (x = x_dead) y-axis offset
      - y_inf: y-axis asymptote (i.e. y(x - x_0 >> tau) => y_inf)
      - tau: decay constant

    Derived parameters:
      - x_1_e: x-axis value for 1/e decay including dead time (`x_1_e = x0 + tau`)
    """

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self, x_scale: float, y_scale: float) -> bool:
        return True

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x_dead: ModelParameter(
            lower_bound=0,
            fixed_to=0,
            scale_func=common.scale_x,
        ),
        y0: ModelParameter(scale_func=common.scale_y),
        y_inf: ModelParameter(scale_func=common.scale_y),
        tau: ModelParameter(lower_bound=0, scale_func=common.scale_x),
    ) -> Array[("num_samples",), np.float64]:
        y = y0 + (y_inf - y0) * (1 - np.exp(-(x - x_dead) / tau))
        y = np.where(x >= x_dead, y, y0)
        return y

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        # Exponentials are generally pretty easy to fit so we keep the estimator simple
        self.parameters["x_dead"].heuristic = 0
        self.parameters["y0"].heuristic = y[0]
        self.parameters["y_inf"].heuristic = y[-1]
        self.parameters["tau"].heuristic = x.ptp()

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {"x_1_e": fitted_params["x_dead"] + fitted_params["tau"]}
        derived_uncertainties = {
            "x_1_e": np.sqrt(
                fit_uncertainties["x_dead"] ** 2 + fit_uncertainties["tau"] ** 2
            )
        }
        return derived_params, derived_uncertainties
