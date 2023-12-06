from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model, ModelParameter
from .utils import PeriodicModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class Ramsey(Model):
    """Fit model for detuning scans of Ramsey experiments (for time scans, use the
    Sinusoid model).

    This model calculates the measurement outcomes for Ramsey experiments, defined by:
        `P = P_readout_g + (P_readout_e - P_readout_g) * P_e`
        where `P_e` is the (time-dependent) population in the excited state and
    `P_readout_g` and `P_readout_e` are the readout levels (measurement outcomes
    when the qubit is in one state).

    The model requires that the system starts out entirely in one of the ground or
    excited states, specified using :meth:`__init__`'s :param:`start_excited` parameter.

    Model parameters:
        - P_readout_e: excited state readout level
        - P_readout_g: ground state readout level
        - t: Ramsey delay
        - t_pi_2: duration of the pi/2 pulses. The pi/2 pulses are assumed to be
            ideal pi/2 pulses with a corresponding Rabi frequency of
            `Omega = np.pi / (2 * t_pi_2)`
        - w_0: resonance frequency offset, defined such that the Ramsey detuning is
            given by `delta = x - w_0`
        - phi: phase of the second pi/2 pulse relative to the first pi/2 pulse
        - tau: decay time constant (fixed to infinity by default)

    Derived parameters:
        - f_0: resonance frequency offset in linear units, given by `w_0 / (2 * np.pi)`

    All frequencies are in angular units.
    """

    def __init__(self, start_excited: bool):
        super().__init__()
        self.start_excited = start_excited

    def get_num_y_channels(self) -> int:
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        P_readout_e: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        P_readout_g: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        t: ModelParameter(
            lower_bound=0.0, scale_func=(lambda x_scale, y_scale, _: 1 / x_scale)
        ),
        t_pi_2: ModelParameter(
            lower_bound=0.0, scale_func=lambda x_scale, y_scale, _: 1 / x_scale
        ),
        w_0: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        phi: PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
        ),
        tau: ModelParameter(
            lower_bound=0.0,
            fixed_to=np.inf,
            scale_func=lambda x_scale, y_scale, _: 1 / x_scale,
        ),
    ):
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

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        if self.start_excited:
            model_parameters["P_readout_e"].heuristic = y[0]
            model_parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            model_parameters["P_readout_g"].heuristic = y[0]
            model_parameters["P_readout_e"].heuristic = abs(1 - y[0])

        unknowns = {
            param_name
            for param_name, param_data in model_parameters.items()
            if not param_data.has_user_initial_value() and param_name != "tau"
        }

        # This is the most common use-case for this model. TODO: cover other use-cases
        # as they arise...
        if unknowns.issubset({"w_0", "phi"}):
            model_parameters["phi"].heuristic = 0.0
            model_parameters["w_0"].heuristic = self.find_x_offset_sampling(
                x=x,
                y=y,
                parameters=model_parameters,
                width=2 * np.pi / model_parameters["t"].get_initial_value(),
                x_offset_param_name="w_0",
            )
        else:
            raise ValueError(
                f"No Ramsey heuristic currently available for the unknowns: {unknowns}"
            )

        model_parameters["tau"].heuristic = (
            10 * model_parameters["t"].get_initial_value()
        )

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        derived_params["f_0"] = fitted_params["w_0"] / (2 * np.pi)
        derived_uncertainties["f_0"] = fit_uncertainties["w_0"] / (2 * np.pi)

        return derived_params, derived_uncertainties
