from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_x, scale_y


class Exponential(Model):
    """Exponential function according to::

        y(x < x_dead) = y0
        y(x >= x_dead) = y0 + (y_inf-y0)*(1-exp(-(x-x_dead)/tau))

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
        x_dead: ModelParameter(
            lower_bound=0,
            fixed_to=0,
            scale_func=scale_x(),
        ),
        y0: ModelParameter(scale_func=scale_y()),
        y_inf: ModelParameter(scale_func=scale_y()),
        tau: ModelParameter(lower_bound=0, scale_func=scale_x()),
    ) -> TY:
        """
        :param x_dead: x-axis "dead time" (fixed to ``0`` by default)
        :param y0: initial (``x = x_dead``) y-axis offset
        :param y_inf: y-axis asymptote (i.e. ``y(x - x_dead >> tau) => y_inf``)
        :param tau: decay constant
        """
        y = y0 + (y_inf - y0) * (1 - np.exp(-(x - x_dead) / tau))
        y = np.where(x >= x_dead, y, y0)
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        # Ensure that y is a 1D array
        x = np.squeeze(x)
        y = np.squeeze(y)

        # Exponentials are generally pretty easy to fit so we keep the estimator simple
        self.parameters["x_dead"].heuristic = 0
        self.parameters["y0"].heuristic = y[0]
        self.parameters["y_inf"].heuristic = y[-1]
        self.parameters["tau"].heuristic = np.ptp(x)

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Derived parameters:

            * ``x_1_e``: x-axis value for ``1/e`` decay including dead time
              (``x_1_e = x_dead + tau``)

        """
        derived_params = {"x_1_e": fitted_params["x_dead"] + fitted_params["tau"]}
        derived_uncertainties = {
            "x_1_e": np.sqrt(
                fit_uncertainties["x_dead"] ** 2 + fit_uncertainties["tau"] ** 2
            )
        }
        return derived_params, derived_uncertainties
