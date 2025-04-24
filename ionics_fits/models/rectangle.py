from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_x, scale_y


class Rectangle(Model):
    """Rectangle function according to::

        x <= x_l: y = y0
        x >= x_r: y = y0
        x_r > x > x_l: y0 + a

    For ``x_l = y0 = 0``, ``x_r = inf`` this is a Heaviside step function.
    """

    def __init__(self, thresh: float = 0.5):
        """
        :param thresh: threshold used to configure the parameter estimator, which
            attempts to find the edges of the rectangle by applying this threshold to
            the peak height above the baseline to determine which points are inside
            / outside the rectangle.
        """
        self.thresh = thresh
        super().__init__()

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
        a: ModelParameter(scale_func=scale_y()),
        y0: ModelParameter(scale_func=scale_y()),
        x_l: ModelParameter(scale_func=scale_x()),
        x_r: ModelParameter(scale_func=scale_x()),
    ) -> TY:
        """
        :param a: rectangle height above the baseline
        :param y0: y-axis offset
        :param x_l: left transition point
        :param- x_r: right transition point
        """
        return np.where(np.logical_and(x_r > x, x > x_l), y0 + a, y0)

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        unknowns = {
            param
            for param, param_data in self.parameters.items()
            if not param_data.has_user_initial_value()
        }

        if {"x_l", "x_r"}.issubset(unknowns):
            self.parameters["y0"].heuristic = 0.5 * (y[0] + y[-1])

        elif "x_l" not in unknowns:
            x_l = self.parameters["x_l"].get_initial_value()

            if min(x) < x_l:
                self.parameters["y0"].heuristic = np.mean(y[x < x_l])
            else:
                y0 = self.parameters["y0"].heuristic = y[-1]
                self.parameters["a"].heuristic = y[0] - y0

        elif "x_r" not in unknowns:
            x_r = self.parameters["x_r"].get_initial_value()
            if max(x) > x_r:
                self.parameters["y0"].heuristic = np.mean(y[x > x_r])
            else:
                y0 = self.parameters["y0"].heuristic = y[0]
                self.parameters["a"].heuristic = y[-1] - y0

        else:
            x_l = self.parameters["x_l"].get_initial_value()
            x_r = self.parameters["x_r"].get_initial_value()

            outside = np.logical_or(x <= x_l, x >= x_r)
            inside = np.logical_and(x > x_l, x < x_r)
            self.parameters["y0"].heuristic = np.mean(y[outside])
            y0 = self.parameters["y0"].get_initial_value()
            self.parameters["a"].heuristic = np.mean(y[outside] - y0)

        y0 = self.parameters["y0"].get_initial_value()
        self.parameters["a"].heuristic = y[np.argmax(np.abs(y - y0))] - y0
        a = self.parameters["a"].get_initial_value()

        thresh = self.thresh * (y0 + (y0 + a))
        inside = (y >= thresh) if a > 0 else (y < thresh)

        if x[inside].size == 0:
            x_l = x[0]
            x_r = x[-1]
        else:
            x_l = min(x[inside])
            x_r = max(x[inside])

        self.parameters["x_l"].heuristic = x_l
        self.parameters["x_r"].heuristic = x_r
