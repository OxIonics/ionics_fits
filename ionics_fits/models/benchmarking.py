from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .. import common, Model, ModelParameter
from ..utils import Array

if TYPE_CHECKING:
    num_samples = float


def _p_scale_func(x_scale: float, y_scale: float) -> float:
    if x_scale != 1.0:
        raise RuntimeError("Benchmarking model cannot be rescaled along x")
    return 1


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
        super().__init__()
        self.parameters["y_inf"].fixed_to = 1 / 2**num_qubits
        self.num_qubits = num_qubits
        self.alpha = 2**num_qubits / (2**num_qubits - 1)

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[bool, bool]:
        return False, True

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        p: ModelParameter(lower_bound=0.0, upper_bound=1.0, scale_func=_p_scale_func),
        y0: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=common.scale_y,
        ),
        y_inf: ModelParameter(
            # fixed_to set to `1 / 2**num_qubits` in the constructor
            lower_bound=0,
            upper_bound=1,
            scale_func=common.scale_y,
        ),
    ) -> Array[("num_samples",), np.float64]:
        y = (y0 - y_inf) * p**x + y_inf
        return y

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        self.parameters["p"].heuristic = 1.0
        self.parameters["y0"].heuristic = max(y)
        self.parameters["y_inf"].heuristic = 1 / 2**self.num_qubits

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
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
