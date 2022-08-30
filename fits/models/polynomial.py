import functools
import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

from . import utils
from .. import FitModel, FitParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float
    num_values = float


class Power(FitModel):
    """Single-power fit according to:
    y = a*(x-x0)^n + y0

    `x - x0` must always be strictly greater than 0. This is because `n` can take
    non-integral values (for integer coefficients use :class Polynomial: instead)
    and this function's return is real-valued.

    The fit will often struggle when both y0 and n are floated if the dataset doesn't
    contain some asymptotic values where `y ~ y0`. The more you can help it out by
    bounding parameters and providing initial guesses the better.

    The fit will generally struggle to converge if both `a` and `y0` are floated unless
    it is given some guidance (e.g. initial values).

    Fit parameters (all floated by default unless stated otherwise):
      - a: y-axis scale factor (fixed to 1 by default)
      - x0: x-axis offset (fixed to 0 by default). This parameter is not expected to be
        used often.
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

    @classmethod
    def func(
        cls, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given
        parameter set and returns the result.

        :param x: x-axis data
        :param params: dictionary of parameter values
        :returns: array of model values
        """
        a = params["a"]
        x0 = params["x0"]
        y0 = params["y0"]
        n = params["n"]

        assert all(x - x0 >= 0), "`x - x0` must be > 0"

        return a * np.float_power(x - x0, n) + y0

    @classmethod
    def estimate_parameters(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the model parameter values for the
        specified dataset.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values, typically called as part of `FitBase.fit`.

        :param x: x-axis data
        :param y: y-axis data
        :param known_values: dictionary mapping model parameter names to values
            parameters whose value is known (e.g. because the parameter is fixed to a
            certain value or an initial value has been provided by the user).
        :param bounds: dictionary of parameter bounds. Estimated values will be clipped
            to lie within bounds.
        """
        param_guesses = dict(known_values)

        def optimal_n():
            """Find the optimal (in the least-squared residuals sense) value of `n`
            based on our current best guesses for the other parameters.

            We use the fact that if `y = x^n` then `n = log(y) / log(x)`. This gives
            us an estimate for `n` at each value of x. We choose the one which results
            in lowest sum of squares residuals.
            """
            if "n" in known_values.keys():
                return known_values["n"], 0

            y0 = param_guesses["y0"]
            a = param_guesses["a"]
            x0 = param_guesses["x0"]

            x_pr = x - x0
            y_pr = y - y0
            y_pr = y_pr / a

            # avoid divide by zero errors
            valid = np.argwhere(np.logical_and(y_pr != 0, x_pr != 1))

            if len(valid) == 0:
                return 0, np.inf

            n = np.log(np.abs(y_pr[valid])) / np.log(np.abs(x_pr[valid]))
            n = n.squeeze()

            # don't look for silly values of n
            n_min = max(-10, bounds["n"][0])
            n_max = min(10, bounds["n"][1])

            n = n[np.argwhere(np.logical_and(n >= n_min, n <= n_max))]

            if len(n) == 0:
                return 1, np.inf

            return cls.param_min_sqrs(x, y, param_guesses, "n", n)

        # We don't have a heuristic for x0
        unknowns = set(["x0", "a", "y0", "n"]) - set(known_values.keys())

        # set some fallbacks for if cases where our heuristics fail us
        param_guesses["x0"] = param_guesses.get("x0", 0)
        param_guesses["y0"] = param_guesses.get("y0", 0)
        param_guesses["a"] = param_guesses.get("a", 1)
        param_guesses["n"] = param_guesses.get("n", 1)

        if len(unknowns) == 0:
            pass  # nothing to do

        elif "x0" in unknowns:
            pass  # don't have a good heuristic for this case

        elif len(unknowns) > 2:
            # Really hard to have good heuristics with this many free parameters
            # unless tight bounds have been set this is likely to fail
            pass

        elif unknowns == set("a"):
            x0 = param_guesses["x0"]
            y0 = param_guesses["y0"]
            n = param_guesses["n"]

            x = x - x0
            y = y - y0
            a = y / np.float_power(x, n)
            a = cls.param_min_sqrs(x, y, param_guesses, "a", a)[0]

            param_guesses["a"] = a

        elif unknowns == set("n"):
            param_guesses["n"] = optimal_n()[0]

        elif unknowns == set("y0"):
            a = param_guesses["a"]
            x0 = param_guesses["x0"]
            n = param_guesses["n"]

            y0 = y - a * np.float_power(x - x0, n)
            y0 = cls.param_min_sqrs(x, y, param_guesses, "y0", y0)[0]

            param_guesses["y0"] = y0

        elif "a" in unknowns:
            pass  # don't have a great heuristic for these cases

        else:
            assert unknowns == set(["y0", "n"])

            # Datasets normally taken such that they contain a value of y close to y0
            y0_guesses = np.array(
                [
                    0,
                    np.min(y),
                    np.max(y),
                    np.mean(y),
                ]
            )
            if all(np.isfinite(bounds["y0"])):
                y0_guesses = y0_guesses.append(bounds["y0"])

            ns = np.zeros_like(y0_guesses)
            costs = np.zeros_like(y0_guesses)
            for idx, y0 in np.ndenumerate(y0_guesses):
                param_guesses["y0"] = y0
                ns[idx], costs[idx] = optimal_n()
            param_guesses["n"] = float(ns[np.argmin(costs)])
            param_guesses["y0"] = float(y0_guesses[np.argmin(costs)])

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
                fixed_to=0,
                scale_func=lambda x_scale, y_scale, fixed_params: 1
                if fixed_params.get("x0") == 0
                else None,
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
      - x0: x-axis offset (fixed to 0 by default). Floating x0 as well as polynomial
          coefficients results in an under-defined problem.

    Derived parameters:
        None
    """

    _POLY_DEGREE = 10
    _PARAMETERS = _generate_poly_parameters(_POLY_DEGREE)

    @classmethod
    def func(
        cls, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given
        parameter set and returns the result.

        :param x: x-axis data
        :param params: dictionary of parameter values
        :returns: array of model values
        """
        x0 = params["x0"]
        p = np.array([params[f"a_{n}"] for n in range(10, -1, -1)], dtype=np.float64)
        assert len(p) == 11

        y = np.polyval(p, x - x0)
        return y

    @classmethod
    def estimate_parameters(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the model parameter values for the
        specified dataset.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values, typically called as part of `FitBase.fit`.

        :param x: x-axis data
        :param y: y-axis data
        :param known_values: dictionary mapping model parameter names to values
            parameters whose value is known (e.g. because the parameter is fixed to a
            certain value or an initial value has been provided by the user).
        :param bounds: dictionary of parameter bounds. Estimated values will be clipped
            to lie within bounds.
        """
        param_guesses = dict(known_values)
        param_guesses["x0"] = param_guesses.get("x0", 0)

        free = [
            n for n in range(cls._POLY_DEGREE + 1) if param_guesses.get(f"a_{n}") != 0.0
        ]
        if len(free) == 0:
            return param_guesses

        deg = max(free)

        p = np.polyfit(x - param_guesses["x0"], y, deg)

        param_guesses.update({f"a_{n}": p[deg - n] for n in range(deg + 1)})

        return param_guesses


line_unused = {f"a_{n}": 0 for n in range(2, 11)}
line_unused.update({"x0": 0})


@utils.rename_params(param_map={"a": "a_1", "y0": "a_0"}, unused_params=line_unused)
class Line(Polynomial):
    """Straight line fit according to:
    `y = a * x + y0`

    Fit parameters (all floated by default unless stated otherwise):
      - y0: y-axis intercept
      - a: slope

    Derived parameters:
        None
    """

    _POLY_DEGREE = 1


parabola_unused = {f"a_{n}": 0 for n in range(3, 11)}
parabola_unused.update({"a_1": 0})


@utils.rename_params(
    param_map={"k": "a_2", "y0": "a_0", "x0": "x0"}, unused_params=parabola_unused
)
class Parabola(Polynomial):
    """Parabola fit according to:
    `y = k * (x - x0)^2 + y0`

    Fit parameters (all floated by default unless stated otherwise):
      - y0: y-axis intercept
      - k: curvature

    Derived parameters:
        None
    """

    _POLY_DEGREE = 2

    @classmethod
    def estimate_parameters(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the model parameter values for the
        specified dataset.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values, typically called as part of `FitBase.fit`.

        Maps the Polynomial `a_1` coefficient onto a value for `x0` according to
        ```
        y = a_0 + a_2 * x^2
        x -> x - x0: y = a_0 + a_2 * (x - x0)^2
        y = a_0 + a_2*x^2 + a_2 * x0^2 + 2*a_2*x*x0
        y = (a_0 + a_2 * x0^2) + 2*a_2*x*x0 + a_2*x^2

        a_0 -> a_0 + a_2 * x0^2
        a_1 -> 2*a_2*x0 => x0 = a_1/(2*a_2)
        a_2 -> a_2
        ```

        :param x: x-axis data
        :param y: y-axis data
        :param known_values: dictionary mapping model parameter names to values
            parameters whose value is known (e.g. because the parameter is fixed to a
            certain value or an initial value has been provided by the user).
        :param bounds: dictionary of parameter bounds. Estimated values will be clipped
            to lie within bounds.
        """
        param_guesses = super().estimate_parameters(x, y, known_values, bounds)

        if "x0" not in known_values:
            a_0 = param_guesses["a_0"]
            a_1 = param_guesses["a_1"]
            a_2 = param_guesses["a_2"]

            param_guesses["x0"] = x0 = -a_1 / (2 * a_2)
            param_guesses["y0"] = known_values.get("y0", a_0 - a_2 * x0**2)

        return param_guesses


Parabola._PARAMETERS["x0"].fixed_to = None
Parabola._PARAMETERS["k"].fixed_to = None
