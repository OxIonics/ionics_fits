from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_invariant, scale_x, scale_x_inv, scale_y
from . import heuristics
from .utils import PeriodicModelParameter


class Ramsey(Model):
    r"""Fit model for detuning scans of Ramsey experiments (for time scans, use the
    Sinusoid model).

    This model calculates the measurement outcomes for Ramsey experiments, defined by::

        P = P_readout_g + (P_readout_e - P_readout_g) * P_e

    where ``P_e`` is the (time-dependent) population in the excited state and
    ``P_readout_g`` and ``P_readout_e`` are the readout levels (measurement outcomes
    when the qubit is in one state).

    The model requires that the system starts out entirely in one of the ground or
    excited states, specified using :meth:``__init__``\'s ``start_excited`` parameter.

    All frequencies are in angular units.
    """

    def __init__(self, start_excited: bool):
        super().__init__()
        self.start_excited = start_excited

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [False]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        P_readout_e: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=scale_y(),
        ),
        P_readout_g: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=scale_y(),
        ),
        t: ModelParameter(
            lower_bound=0.0,
            scale_func=scale_x_inv(),
        ),
        t_pi_2: ModelParameter(
            lower_bound=0.0,
            scale_func=scale_x_inv(),
        ),
        w_0: ModelParameter(scale_func=scale_x()),
        phi: PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
            scale_func=scale_invariant,
        ),
        tau: ModelParameter(
            lower_bound=0.0,
            fixed_to=np.inf,
            scale_func=scale_x_inv(),
        ),
    ):
        """
        :param P_readout_e: excited state readout level
        :param P_readout_g: ground state readout level
        :param t: Ramsey delay
        :param t_pi_2: duration of the pi/2 pulses. The pi/2 pulses are assumed to be
            ideal pi/2 pulses with a corresponding Rabi frequency of
            ``Omega = np.pi / (2 * t_pi_2)``
        :param w_0: resonance frequency offset, defined such that the Ramsey detuning is
            given by ``delta = x - w_0``
        :param phi: phase of the second pi/2 pulse relative to the first pi/2 pulse
        :param tau: decay time constant (fixed to infinity by default)
        """
        delta = x - w_0
        Omega = np.pi / (2 * t_pi_2)

        alpha = (delta / Omega) ** 2
        theta = np.pi / 4 * np.sqrt(1 + alpha)
        gamma = np.arctan(np.sqrt(alpha / (1 + alpha)) * np.tan(theta))

        P_trans = (
            2
            * np.exp(-t / tau)
            / (1 + alpha)
            * np.sin(theta) ** 2
            * (np.cos(theta) ** 2 + alpha / (1 + alpha) * np.sin(theta) ** 2)
            * (1 + np.cos(delta * t + phi + 2 * gamma))
        )

        P_e = 1 - P_trans if self.start_excited else P_trans
        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        if self.start_excited:
            self.parameters["P_readout_e"].heuristic = y[0]
            self.parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            self.parameters["P_readout_g"].heuristic = y[0]
            self.parameters["P_readout_e"].heuristic = abs(1 - y[0])

        unknowns = {
            param_name
            for param_name, param_data in self.parameters.items()
            if not param_data.has_user_initial_value() and param_name != "tau"
        }

        # This is the most common use-case for this model. TODO: cover other use-cases
        # as they arise...
        if unknowns.issubset({"w_0", "phi"}):
            self.parameters["phi"].heuristic = 0.0
            self.parameters["w_0"].heuristic = heuristics.find_x_offset_sampling(
                model=self,
                x=x,
                y=y,
                width=2 * np.pi / self.parameters["t"].get_initial_value(),
                x_offset_param_name="w_0",
            )
        else:
            raise ValueError(
                f"No Ramsey heuristic currently available for the unknowns: {unknowns}"
            )

        self.parameters["tau"].heuristic = 10 * self.parameters["t"].get_initial_value()

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Derived parameters:

        * ``f_0``: resonance frequency offset in linear units, given by
          ``w_0 / (2 * np.pi)``
        """
        derived_params = {}
        derived_uncertainties = {}

        derived_params["f_0"] = fitted_params["w_0"] / (2 * np.pi)
        derived_uncertainties["f_0"] = fit_uncertainties["w_0"] / (2 * np.pi)

        return derived_params, derived_uncertainties
