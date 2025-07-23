from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..normal import NormalFitter
from ..utils import scale_x, scale_y
from .polynomial import Line


class Staircase(Model):
    """Staircase function according to::

    with the nearest integer function := nint()
    y = a * nint( (x - x0) / w) + y0

    Model parameters:
    * ``a``: step size
    * ``w``: step width
    * ``x0``: staircase "offset" in x
    * ``y0``: staircase "offset" in y. By default fixed to 0.
    Floating ``x0`` as well as ``y0`` results in an under-defined problem.
    """

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [True]

    def _func(
        self,
        x: TX,
        a: ModelParameter(scale_func=scale_y()),
        w: ModelParameter(
            scale_func=scale_x(), lower_bound=0
        ),  # is that the right scale function?
        x0: ModelParameter(scale_func=scale_x()),
        y0: ModelParameter(fixed_to=0, scale_func=scale_y()),
    ) -> TY:
        return a * np.round((x - x0) / w) + y0

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        self.parameters["a"].heuristic = max(y[1:] - y[:-1])

        x_edges = x[np.where(y[1:] - y[:-1] > (self.parameters["a"].heuristic / 2))]
        if len(x_edges) > 1:
            w_guess = np.mean(x_edges[1:] - x_edges[:-1])
        elif len(x_edges) == 1:
            w_guess = max([x_edges[0] - x[0], x[-1] - x_edges[0]])
        else:
            w_guess = x[-1] - x[0]
        self.parameters["w"].heuristic = w_guess

        if (self.parameters["x0"].fixed_to is None) and (
            self.parameters["y0"].fixed_to is not None
        ):
            fit = NormalFitter(
                x, y - self.parameters["y0"].get_initial_value(), model=Line()
            )
            self.parameters["x0"].heuristic = -fit.values["y0"] / fit.values["a"]
        elif (self.parameters["x0"].fixed_to is not None) and (
            self.parameters["y0"].fixed_to is None
        ):
            self.parameters["y0"].heuristic = np.mean(
                y
                - self.parameters["a"].get_initial_value()
                * np.round(
                    (x - self.parameters["x0"].get_initial_value())
                    / self.parameters["w"].get_initial_value()
                )
            )
        else:
            raise Exception(
                "``x0`` and ``y0`` cannot be floated simultaneously in Staircase"
                "function"
            )
