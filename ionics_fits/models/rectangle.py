from typing import Dict, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


class Rectangle(Model):
    """Rectangle function according to:
    x <= x_l: y = y0
    x >= x_r: y = y0
    x_r > x > x_l: y0 + a

    Fit parameters (all floated by default unless stated otherwise):
      - a: rectangle height above the baseline
      - y0: y-axis offset
      - x_l: left transition point
      - x_r: right transition point

    Derived parameters:
      None

    For `x_l = y0 = 0`, `x_r = inf` this is a Heaviside step function.
    """

    def __init__(self, thresh: float = 0.5):
        """threshold is used to configure the parameter estimator"""
        self.thresh = thresh
        super().__init__()

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        a: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        x_l: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        x_r: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
    ) -> Array[("num_samples",), np.float64]:
        return np.where(np.logical_and(x_r > x, x > x_l), y0 + a, y0)

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

        unknowns = {
            param
            for param, param_data in model_parameters.items()
            if not param_data.has_user_initial_value()
        }

        if {"x_l", "x_r"}.issubset(unknowns):
            model_parameters["y0"].heuristic = 0.5 * (y[0] + y[-1])

        elif "x_l" not in unknowns:
            x_l = model_parameters["x_l"].get_initial_value()

            if min(x) < x_l:
                model_parameters["y0"].heuristic = np.mean(y[x < x_l])
            else:
                y0 = model_parameters["y0"].heuristic = y[-1]
                model_parameters["a"].heuristic = y[0] - y0

        elif "x_r" not in unknowns:
            x_r = model_parameters["x_r"].get_initial_value()
            if max(x) > x_r:
                model_parameters["y0"].heuristic = np.mean(y[x > x_r])
            else:
                y0 = model_parameters["y0"].heuristic = y[0]
                model_parameters["a"].heuristic = y[-1] - y0

        else:
            x_l = model_parameters["x_l"].get_initial_value()
            x_r = model_parameters["x_r"].get_initial_value()

            outside = np.logical_or(x <= x_l, x >= x_r)
            inside = np.logical_and(x > x_l, x < x_r)
            model_parameters["y0"].heuristic = np.mean(y[outside])
            y0 = model_parameters["y0"].get_initial_value()
            model_parameters["a"].heuristic = np.mean(y[outside] - y0)

        y0 = model_parameters["y0"].get_initial_value()
        model_parameters["a"].heuristic = y[np.argmax(np.abs(y - y0))] - y0
        a = model_parameters["a"].get_initial_value()

        thresh = self.thresh * (y0 + (y0 + a))
        inside = (y >= thresh) if a > 0 else (y < thresh)

        if x[inside].size == 0:
            x_l = x[0]
            x_r = x[-1]
        else:
            x_l = min(x[inside])
            x_r = max(x[inside])

        model_parameters["x_l"].heuristic = x_l
        model_parameters["x_r"].heuristic = x_r
