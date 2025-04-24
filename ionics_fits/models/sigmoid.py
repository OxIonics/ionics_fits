from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_x, scale_x_inv, scale_y


class LogisticFunction(Model):
    """Logistic function model according to::

        y = a / (1 + exp(-k*(x - x0))) + y0

    See :meth:`_func` for parameters.
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
        a: ModelParameter(scale_func=scale_y()),
        y0: ModelParameter(scale_func=scale_y()),
        x0: ModelParameter(scale_func=scale_x()),
        k: ModelParameter(scale_func=scale_x_inv(), lower_bound=0),
    ) -> TY:
        """
        :param a: y-axis scale factor
        :param y0: y-axis offset
        :param x0: x-axis offset
        :param k: logistic growth rate (steepness of the curve)
        """
        return a / (1 + np.exp(-k * (x - x0))) + y0

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        sgn = +1 if y[-1] > y[0] else -1
        y0 = self.parameters["y0"].get_initial_value(y[0])
        self.parameters["y0"].heuristic = y0

        a = self.parameters["a"].get_initial_value(sgn * np.ptp(y))
        self.parameters["a"].heuristic = a

        y_centre = y0 + a / 2
        x0 = self.parameters["x0"].get_initial_value(x[np.argmin(np.abs(y - y_centre))])
        self.parameters["x0"].heuristic = x0

        FWHMH_pts = np.abs(y - y_centre) < np.abs(a / 4)
        x_FWHMH = x[FWHMH_pts]
        FWHMH = np.ptp(x_FWHMH)

        k = self.parameters["k"].get_initial_value(1.098 / (FWHMH / 2))
        self.parameters["k"].heuristic = k
