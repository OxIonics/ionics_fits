import functools
import numpy as np
from typing import Dict

from .. import FitModel, FitParameter


class Power(FitModel):
    """Single-power fit according to:
    y = a*(x-x0)^n + y0

    `x - x0` must always be strictly greater than 0. This is because (a) `n` is
    generally not an integer (for integer coefficients use :class Polynomial: instead)
    and (b) this function only returns real-valued numbers.

    Fit parameters (all floated by default unless stated otherwise):
      - a: y-axis scale factor (fixed to 1 by default)
      - x0: x-axis offset (fixed to 0 by default)
      - y0: y-axis offset
      - n: power

    Derived parameters:
        None
    """

    _PARAMETERS: Dict[str, FitParameter] = {
        "a": FitParameter(
            fixed_to=1,
            scale_func=lambda x_scale, y_scale, fixed_params: None
            if "n" not in fixed_params
            else y_scale / np.float_power(x_scale, fixed_params["n"]),
        ),
        "x0": FitParameter(fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale),
        "y0": FitParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        "n": FitParameter(),
    }

    @staticmethod
    def func(x: np.array, params: Dict[str, float]) -> np.array:
        """Returns the model function values at the points specified by `x` for the
        parameter values specified by `params`.
        """
        if any(x < 0):
            raise ValueError("Negative x is currently not supported")

        a = params["a"]
        x0 = params["x0"]
        y0 = params["y0"]
        n = params["n"]

        if any(x - x0 <= 0):
            raise ValueError("`x - x0` must be > 0")

        return a * np.float_power(x - x0, n) + y0

    @staticmethod
    def estimate_parameters(x, y, known_values, bounds) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the parameter values for the specified
        dataset.

        The dataset must be sorted in order of increasing x-axis values and not contain
        any infinite or nan values.

        :param x: dataset x-axis values
        :param y: dataset y-axis values
        :param known_values: dictionary of parameters whose value is known (e.g. because
            the parameter is fixed to a certain value or an estimate guess has been
            provided by the user).
        :param bounds: dictionary of parameter bounds. All estimated values must lie
            within the specified parameter bounds.
        """
        param_guesses = {}
        param_guesses["x0"] = known_values.get("x0", 0)
        param_guesses["n"] = known_values.get("n", 1)
        param_guesses["y0"] = known_values.get("y0", 0)
        param_guesses["a"] = known_values.get("a", 1)
        return param_guesses


def _generate_poly_parameters(poly_degree):
    def scale_func(n, x_scale, y_scale, _):
        return y_scale / np.power(x_scale, n)

    params = {
        f"a_{n}": FitParameter(
            fixed_to=None if n <= 1 else 0,
            scale_func=functools.partial(scale_func, n),
        )
        for n in range(11)
    }
    params.update(
        {
            "x0": FitParameter(
                fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale
            ),
        }
    )
    return params


class Polynomial(FitModel):
    """10th-order polynomial fit according to:
    y = sum(a_n*(x-x0)^n) for n ={0...10}

    Fit parameters (all floated by default unless stated otherwise):
      - a_0 ... a_10: polynomial coefficients. All polynomial coefficients above 1 are
          fixed to 0 by default.
      - x0: x-axis offset (fixed to 0 by default). NB Floating x0 as well as the
          polynomial coefficients results in a rather under-defined problem...use with
          care and provide initial parameter guesses where possible!

    Derived parameters:
        None
    """

    _POLY_DEGREE = 10
    _PARAMETERS = _generate_poly_parameters(_POLY_DEGREE)

    @staticmethod
    def func(x: np.array, params: Dict[str, float]) -> np.array:
        """Returns the model function values at the points specified by `x` for the
        parameter values specified by `params`.
        """
        x0 = params["x0"]
        p = np.array([params[f"a_{n}"] for n in range(10, -1, -1)], dtype=np.float64)
        assert len(p) == 11

        y = np.polyval(p, x - x0)
        return y

    @staticmethod
    def estimate_parameters(x, y, known_values, bounds) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the parameter values for the specified
        dataset.

        The dataset must be sorted in order of increasing x-axis values and not contain
        any infinite or nan values.

        :param x: dataset x-axis values
        :param y: dataset y-axis values
        :param known_values: dictionary of parameters whose value is known (e.g. because
            the parameter is fixed to a certain value or an estimate guess has been
            provided by the user).
        :param bounds: dictionary of parameter bounds. All estimated values must lie
            within the specified parameter bounds.
        """
        param_guesses = dict(known_values)
        param_guesses["x0"] = param_guesses.get("x0", 0)

        free = [
            n
            for n in range(Polynomial._POLY_DEGREE + 1)
            if param_guesses.get(f"a_{n}") != 0.0
        ]
        if len(free) == 0:
            return param_guesses

        deg = max(free)
        p = np.polyfit(x - param_guesses["x0"], y, deg)

        param_guesses.update(
            {f"a_{n}": param_guesses.get(f"a_{n}", p[deg - n]) for n in range(deg + 1)}
        )

        return param_guesses
