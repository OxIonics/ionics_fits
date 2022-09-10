from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
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

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x_dead: ModelParameter(
            lower_bound=0, fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale
        ),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        y_inf: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        tau: ModelParameter(lower_bound=0, scale_func=lambda x_scale, y_scale, _: None),
    ) -> Array[("num_samples",), np.float64]:
        y = y0 + (y_inf - y0) * (1 - np.exp(-(x - x_dead) / tau))
        y = np.where(x >= x_dead, y, y0)
        return y

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Sets initial values for model parameters based on heuristics. Typically
        called during `Fitter.fit`.

        Heuristic results should stored in :param model_parameters: using the
        `ModelParameter`'s `initialise` method. This ensures that all information passed
        in by the user (fixed values, initial values, bounds) is used correctly.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        # Exponentials are generally pretty easy to fit so we keep the estimator simple
        model_parameters["x_dead"].initialise(0)
        model_parameters["y0"].initialise(y[0])
        model_parameters["y_inf"].initialise(y[-1])
        model_parameters["tau"].initialise(x.ptp())

    @staticmethod
    def calculate_derived_params(
        fitted_params: Dict[str, float], fit_uncertainties: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        derived_params = {"x_1_e": fitted_params["x_dead"] + fitted_params["tau"]}
        derived_uncertainties = {
            "x_1_2": np.sqrt(
                fit_uncertainties["x_dead"] ** 2 + (fit_uncertainties["tau"] / 2) ** 2
            )
        }
        return derived_params, derived_uncertainties
