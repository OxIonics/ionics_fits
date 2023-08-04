from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


def _scale_func(x_scale, y_scale, model):
    # Prevent rescaling
    return None


def _generate_benchmarking_parameters(num_qubits):
    params = {
        "p": ModelParameter(lower_bound=0.0, upper_bound=1.0, scale_func=_scale_func),
        "y0": ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        "y_inf": ModelParameter(
            fixed_to=1 / 2**num_qubits,
            lower_bound=0,
            upper_bound=1,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
    }
    return params


class Benchmarking(Model):
    """Benchmarking success probability decay model

    y = (y0 - y_inf)*p^x + y_inf
    for sequence length x.

    Fit parameters (all floated by default unless stated otherwise):
      - p: depolarisation parameter
      - y0: SPAM fidelity estimate
      - y_inf: depolarisation offset (y-axis asymptote) (fixed to 1/2^n by default)

    Derived parameters:
      - e: error per Clifford = (1 - p) / alpha_n where alpha_n = 2^n / (2^n - 1)
      - e_spam: estimated SPAM error = 1 - y0
    """

    def __init__(self, num_qubits):
        """Init

        :param num_qubits: The number of qubits involved in the benchmarking sequence.
        """
        self.num_qubits = num_qubits
        self.alpha = 2**num_qubits / (2**num_qubits - 1)
        super().__init__(parameters=_generate_benchmarking_parameters(num_qubits))

    def get_num_y_channels(self) -> int:
        """Return the number of y-channels supported by the model."""
        return 1

    def func(
        self, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given
        parameter set and returns the result."""
        p = params["p"]
        y0 = params["y0"]
        y_inf = params["y_inf"]
        y = (y0 - y_inf) * p**x + y_inf
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

        model_parameters["p"].heuristic = 1.0
        model_parameters["y0"].heuristic = max(y)
        model_parameters["y_inf"].heuristic = 1 / 2**self.num_qubits

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
        p = fitted_params["p"]
        y0 = fitted_params["y0"]

        e = (1 - p) / self.alpha
        e_spam = 1 - y0

        p_err = fit_uncertainties["p"]
        y0_err = fit_uncertainties["y0"]

        derived_params = {}
        derived_params["e"] = e
        derived_params["e_spam"] = e_spam

        derived_uncertainties = {}
        derived_uncertainties["e"] = p_err / self.alpha
        derived_uncertainties["e_spam"] = y0_err

        return derived_params, derived_uncertainties
