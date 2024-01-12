from typing import Tuple, TYPE_CHECKING
import numpy as np

from .. import common, Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


class LogisticFunction(Model):
    """Logistic function model according to:
    y = a / (1 + exp(-k*(x - x0))) + y0

    Fit parameters (all floated by default unless stated otherwise):
      - a: y-axis scale factor
      - y0: y-axis offset
      - x0: x-axis offset
      - k: logistic growth rate (steepness of the curve)

    Derived parameters:
      None
    """

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[bool, bool]:
        return True, True

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        a: ModelParameter(scale_func=common.scale_y),
        y0: ModelParameter(scale_func=common.scale_y),
        x0: ModelParameter(scale_func=common.scale_x),
        k: ModelParameter(scale_func=common.scale_x_inv, lower_bound=0),
    ) -> Array[("num_samples",), np.float64]:
        return a / (1 + np.exp(-k * (x - x0))) + y0

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        y = y.squeeze()
        sgn = +1 if y[-1] > y[0] else -1
        y0 = self.parameters["y0"].get_initial_value(y[0])
        self.parameters["y0"].heuristic = y0

        a = self.parameters["a"].get_initial_value(sgn * y.ptp())
        self.parameters["a"].heuristic = a

        y_centre = y0 + a / 2
        x0 = self.parameters["x0"].get_initial_value(x[np.argmin(np.abs(y - y_centre))])
        self.parameters["x0"].heuristic = x0

        FWHMH_pts = np.abs(y - y_centre) < np.abs(a / 4)
        x_FWHMH = x[FWHMH_pts]
        FWHMH = x_FWHMH.ptp()

        k = self.parameters["k"].get_initial_value(1.098 / (FWHMH / 2))
        self.parameters["k"].heuristic = k