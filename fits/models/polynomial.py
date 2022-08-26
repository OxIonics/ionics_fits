import numpy as np
from typing import Dict

from .. import FitModel, FitParameter


class Power(FitModel):
    """Single-power fit according to:
    y = (x-x0)^n + y0

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset (fixed to 0 by default)
      - y0: y-axis offset
      - n: power

    Derived parameters:
        None
    """

    _PARAMETERS: Dict[str, FitParameter] = {
        "x0": FitParameter(fixed_to=0, scale_func=lambda x_scale, y_scale: 1 / x_scale),
        "y0": FitParameter(scale_func=lambda x_scale, y_scale: 1 / y_scale),
        "n": FitParameter(),
    }

    @staticmethod
    def func(x: np.array, params: Dict[str, float]) -> np.array:
        """Returns the model function values at the points specified by `x` for the
        parameter values specified by `params`.
        """
        x0 = params["x0"]
        y0 = params["y0"]
        n = params["n"]
        return np.pow(x - x0, n) + y0

    @staticmethod
    def estimate_parameters(x, y, y_err, known_values, bounds) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the parameter values for the specified
        dataset.

        The dataset must be sorted in order of increasing x-axis values and not contain
        any infinite or nan values.

        :param x: dataset x-axis values
        :param y: dataset y-axis values
        :param y_err: dataset y-axis uncertainties
        :param known_values: dictionary of parameters whose value is known (e.g. because
            the parameter is fixed to a certain value or an estimate guess has been
            provided by the user).
        :param bounds: dictionary of parameter bounds. All estimated values must lie
            within the specified parameter bounds.
        """
        # A single-power polynomial is generally pretty easy to fit so we don't bother
        # to try to do anything clever
        param_guesses = {}
        param_guesses["x0"] = known_values.get("x0", 0)
        param_guesses["n"] = known_values.get("n", 1)
        param_guesses["y0"] = known_values.get("y0", 0)
        return param_guesses
