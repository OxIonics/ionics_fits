from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_no_rescale


class Benchmarking(Model):
    """Benchmarking success probability decay model according to::

        y = (y0 - y_inf)*p^x + y_inf

    where ``x`` is the sequence length (number of Clifford operations).

    See :meth:`_func` for parameter details.
    """

    def __init__(self, num_qubits):
        """
        :param num_qubits: The number of qubits involved in the benchmarking sequence.
        """
        super().__init__()
        self.parameters["y_inf"].fixed_to = 1 / 2**num_qubits
        self.num_qubits = num_qubits
        self.alpha = 2**num_qubits / (2**num_qubits - 1)

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [False], [False]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        p: ModelParameter(
            lower_bound=0.0, upper_bound=1.0, scale_func=scale_no_rescale
        ),
        y0: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=scale_no_rescale,
        ),
        y_inf: ModelParameter(
            # fixed_to set to `1 / 2**num_qubits` in the constructor
            lower_bound=0,
            upper_bound=1,
            scale_func=scale_no_rescale,
        ),
    ) -> TY:
        """Fit parameters

        :param p: depolarisation parameter
        :param y0: SPAM fidelity estimate
        :param y_inf: depolarisation offset (y-axis asymptote) (fixed to ``1/2^n`` by
          default)
        """
        y = (y0 - y_inf) * p**x + y_inf
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        y = np.squeeze(y)

        self.parameters["p"].heuristic = 1.0
        self.parameters["y0"].heuristic = max(y)
        self.parameters["y_inf"].heuristic = 1 / 2**self.num_qubits

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Derived parameters:

        * ``e``: error per Clifford ``e = (1 - p) / alpha_n`` where
          ``alpha_n = 2^n / (2^n - 1)``
        * ``e_spam``: estimated SPAM error ``e_spam = 1 - y0``
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
