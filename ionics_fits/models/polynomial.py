from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from .containers import MappedModel
from .. import Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


def _power_x_scale(x_scale: float, y_scale: float, model: Model) -> Optional[float]:
    if model.parameters["n"].fixed_to is None:
        return None
    return y_scale / np.float_power(x_scale, model.parameters["n"].fixed_to)


class Power(Model):
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

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        a: ModelParameter(fixed_to=1, scale_func=_power_x_scale),
        x0: ModelParameter(fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        n: ModelParameter(),
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result."""
        assert all(x - x0 >= 0), "`x - x0` must be > 0"
        return a * np.float_power(x - x0, n) + y0

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
            if param_data.fixed_to is None
        }

        model_parameters["x0"].heuristic = 0.0
        model_parameters["y0"].heuristic = 0.0
        model_parameters["a"].heuristic = 1.0
        model_parameters["n"].heuristic = 1.0

        x0 = model_parameters["x0"].get_initial_value()
        y0 = model_parameters["y0"].get_initial_value()
        a = model_parameters["a"].get_initial_value()
        n = model_parameters["n"].get_initial_value()

        if len(unknowns) == 0 or "x0" in unknowns or len(unknowns) > 2:
            # No heuristic needed / we don't have a good heuristic for this case
            # Fit may well fail if we hit this :(
            pass

        elif unknowns == {"a"}:
            raise ValueError
            x = x - x0
            y = y - y0
            a = y / np.float_power(x, n)
            a = self.param_min_sqrs(x, y, model_parameters, "a", a)[0]
            model_parameters["a"].heuristic = a

        elif unknowns == {"n"}:
            n = self.optimal_n(x, y, model_parameters)[0]
            model_parameters["n"].heuristic = n

        elif unknowns == {"y0"}:
            y0 = y - a * np.float_power(x - x0, n)
            y0 = self.param_min_sqrs(x, y, model_parameters, "y0", y0)[0]
            model_parameters["y0"].heuristic = y0

        elif "a" in unknowns:
            pass  # don't have a great heuristic for these cases

        else:
            assert unknowns == {"y0", "n"}, unknowns

            # Datasets normally taken such that they contain a value of y close to y0
            y0_guesses = np.array(
                [
                    0,
                    np.min(y),
                    np.max(y),
                    np.mean(y),
                ]
            )
            y0_bounds = [
                model_parameters["y0"].lower_bound,
                model_parameters["y0"].upper_bound,
            ]
            y0_bounds = [bound for bound in y0_bounds if np.isfinite(bound)]
            y0_guesses = np.append(y0_guesses, y0_bounds)

            ns = np.zeros_like(y0_guesses)
            costs = np.zeros_like(y0_guesses)
            for idx, y0 in np.ndenumerate(y0_guesses):
                model_parameters["y0"].heuristic = y0
                ns[idx], costs[idx] = self.optimal_n(x, y, model_parameters)

            model_parameters["n"].heuristic = float(ns[np.argmin(costs)])
            model_parameters["y0"].heuristic = float(y0_guesses[np.argmin(costs)])

    def optimal_n(self, x, y, model_parameters):
        """Find the optimal (in the least-squared residuals sense) value of `n`
        based on our current best guesses for the other parameters.

        We use the fact that if `y = x^n` then `n = log(y) / log(x)`. This gives
        us an estimate for `n` at each value of x. We choose the one which results
        in lowest sum of squares residuals.
        """
        if model_parameters["n"].has_user_initial_value():
            return model_parameters["n"].get_initial_value(), 0

        x0 = model_parameters["x0"].get_initial_value()
        y0 = model_parameters["y0"].get_initial_value()
        a = model_parameters["a"].get_initial_value()

        x_pr = x - x0
        y_pr = y - y0
        y_pr = y_pr / a

        # Avoid divide by zero errors
        valid = np.argwhere(np.logical_and(y_pr != 0, x_pr != 1))

        if len(valid) == 0:
            return 0, np.inf

        n = np.log(np.abs(y_pr[valid])) / np.log(np.abs(x_pr[valid]))
        n = n.squeeze()

        # Don't look for silly values of n
        n_min = max(-10, model_parameters["n"].lower_bound)
        n_max = min(10, model_parameters["n"].upper_bound)

        n = n[np.argwhere(np.logical_and(n >= n_min, n <= n_max))]

        if len(n) == 0:
            return 1, np.inf

        return self.param_min_sqrs(x, y, model_parameters, "n", n)


def poly_fit_parameter(n):
    def scale_func(x_scale, y_scale, _):
        return y_scale / np.power(x_scale, n)

    return ModelParameter(
        fixed_to=None,
        scale_func=scale_func,
    )


def _poly_x_scale(x_scale: float, y_scale: float, model: Model) -> Optional[float]:
    # If a non-zero value has been set for `x0` we can't rescale the dataset
    if model.parameters["x0"].fixed_to == 0.0:
        return 1
    return None


def _generate_poly_parameters(poly_degree):
    params = {f"a_{n}": poly_fit_parameter(n) for n in range(poly_degree + 1)}
    params.update({"x0": ModelParameter(fixed_to=0, scale_func=_poly_x_scale)})
    return params


class Polynomial(Model):
    """A polynomial fit model.

    Fits the function:
    y = sum(a_n*(x-x0)^n) for n ={0...poly_degree}

    Fit parameters (all floated by default unless stated otherwise):
      - a_0 ... a_{poly_degree}: polynomial coefficients.
      - x0: x-axis offset (fixed to 0 by default). Floating x0 as well as polynomial
          coefficients results in an under-defined problem.

    Derived parameters:
        None
    """

    def __init__(self, poly_degree=10):
        """Init

        :param poly_degree: The degree of the polynomial that we're fitting
           defaults to 10.
        """
        self.poly_degree = poly_degree
        super().__init__(parameters=_generate_poly_parameters(poly_degree))

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    def func(
        self, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given
        parameter set and returns the result."""
        x0 = params["x0"]
        p = np.array(
            [params[f"a_{n}"] for n in range(self.poly_degree, -1, -1)],
            dtype=np.float64,
        )

        y = np.polyval(p, x - x0)
        return y

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

        model_parameters["x0"].heuristic = 0.0
        x0 = model_parameters["x0"].get_initial_value()

        free = [
            n
            for n in range(self.poly_degree + 1)
            if model_parameters[f"a_{n}"].fixed_to != 0.0
        ]
        if len(free) == 0:
            return

        deg = max(free)

        p = np.polyfit(x - x0, y, deg)
        for idx in range(deg + 1):
            model_parameters[f"a_{idx}"].heuristic = p[deg - idx]


class Line(MappedModel):
    """Straight line fit according to:
    `y = a * x + y0`

    Fit parameters (all floated by default unless stated otherwise):
      - y0: y-axis intercept
      - a: slope

    Derived parameters:
        None
    """

    def __init__(self):
        super().__init__(
            Polynomial(1),
            {"a": "a_1", "y0": "a_0"},
            {"x0": 0},
        )


class Parabola(MappedModel):
    """Parabola fit according to:
    `y = k * (x - x0)^2 + y0`

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis intercept
      - k: curvature

    Derived parameters:
        None
    """

    def __init__(self):
        inner = Polynomial(2)
        inner.parameters["x0"].fixed_to = None
        inner.parameters["a_2"].fixed_to = None
        super().__init__(
            inner,
            {"k": "a_2", "y0": "a_0", "x0": "x0"},
            {"a_1": 0},
        )

    def _inner_estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        inner_parameters: Dict[str, ModelParameter],
    ):
        """
        If `x0` is floated, we map the Polynomial `a_1` coefficient onto a value for
        `x0` according to:
        ```
        y = a_0 + a_2 * x^2
        x -> x - x0: y = a_0 + a_2 * (x - x0)^2
        y = a_0 + a_2*x^2 + a_2 * x0^2 + 2*a_2*x*x0
        y = (a_0 + a_2 * x0^2) + 2*a_2*x*x0 + a_2*x^2

        a_0 -> a_0 + a_2 * x0^2
        a_1 -> 2*a_2*x0 => x0 = a_1/(2*a_2)
        a_2 -> a_2
        ```
        """
        super()._inner_estimate_parameters(x, y, inner_parameters)

        a_0 = inner_parameters["a_0"].get_initial_value()
        a_1 = inner_parameters["a_1"].heuristic
        a_2 = inner_parameters["a_2"].get_initial_value()

        x0 = -a_1 / (2 * a_2)
        inner_parameters["x0"].heuristic = x0
        inner_parameters["a_0"].heuristic = a_0 - a_2 * x0**2
