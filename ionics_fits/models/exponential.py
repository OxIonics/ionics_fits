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

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x_dead: ModelParameter(
            lower_bound=0, fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale
        ),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        y_inf: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        tau: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: x_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        y = y0 + (y_inf - y0) * (1 - np.exp(-(x - x_dead) / tau))
        y = np.where(x >= x_dead, y, y0)
        return y

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
        # Exponentials are generally pretty easy to fit so we keep the estimator simple
        model_parameters["x_dead"].heuristic = 0
        model_parameters["y0"].heuristic = y[0]
        model_parameters["y_inf"].heuristic = y[-1]
        model_parameters["tau"].heuristic = x.ptp()

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param x: x-axis data
        :param y: y-axis data
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


class Benchmarking(Exponential):
    """Randomised benchmarking success probability decay model

    y = A*p^x + B
    for sequence length x.
    Change basis -> y = A*exp(-(ln(1 / p))*x) + B
    Exponential function with
      y_inf = B
      y0 - y_inf = A
      x_dead = 0
      tau = 1 / (ln(1 / p))

    Fit parameters (all floated by default unless stated otherwise):
      - x_dead: x-axis "dead time" (fixed to 0 by default)
      - y0: SPAM fidelity estimate (initial y-axis offset)
      - y_inf: depolarisation offset (y-axis asymptote)
        (fixed to 1 / 2**num_qubits by default)
      - tau: exponential decay constant

    Derived parameters:
      - A: y0 - y_inf
      - B: y_inf
      - p: depolarisation parameter = exp(-(1 / tau))
      - e: error per Clifford = (1 - p) / alpha_n where alpha_n = 2^n / (2^n - 1)
      - e_spam: estimated SPAM error = 1 - y0
    """

    def __init__(self, num_qubits):
        """Init

        :param num_qubits: The number of qubits involved in the benchmarking sequence
        """
        self.num_qubits = num_qubits
        self.alpha = 2**num_qubits / (2**num_qubits - 1)
        super().__init__()
        self.parameters["y_inf"].fixed_to = 1 / 2**num_qubits
        for param in (self.parameters["y0"], self.parameters["y_inf"]):
            param.lower_bound = 0
            param.upper_bound = 1
        self.parameters["tau"].lower_bound = 1  # Equivalent to p <= 1

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param x: x-axis data
        :param y: y-axis data
        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        y0 = fitted_params["y0"]
        y_inf = fitted_params["y_inf"]
        tau = fitted_params["tau"]

        p = np.exp(-(1 / tau))
        e = (1 - p) / self.alpha

        derived_params = {}
        derived_params["A"] = y0 - y_inf
        derived_params["B"] = y_inf
        derived_params["p"] = p
        derived_params["e"] = e
        derived_params["e_spam"] = 1 - y0

        y0_err = fit_uncertainties["y0"]
        y_inf_err = fit_uncertainties["y_inf"]
        tau_err = fit_uncertainties["tau"]

        derived_uncertainties = {}
        derived_uncertainties["A"] = np.sqrt(y0_err**2 + y_inf_err**2)
        derived_uncertainties["B"] = y_inf_err
        p_err = derived_uncertainties["p"] = (
            (1 / tau**2) * np.exp(-(1 / tau)) * tau_err
        )
        derived_uncertainties["e"] = p_err / self.alpha
        derived_uncertainties["e_spam"] = y0_err

        return derived_params, derived_uncertainties
