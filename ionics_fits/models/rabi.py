from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .sinc import Sinc2
from .sinusoid import Sinusoid
from .. import Model, ModelParameter, NormalFitter
from .utils import get_spectrum
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class RabiFlop(Model):
    """Base class for damped Rabi flopping.

    This model calculates measurement outcomes for two-state systems undergoing damped
    Rabi oscillations, defined by:
        `P = P_readout_g + (P_readout_e - P_readout_g) * P_e`
    where `P_e` is the (time-dependent) population in the excited state and
    `P_readout_g` and `P_readout_e` are the readout levels (measurement outcomes
    when the qubit is in one state).

    This class does not support fitting directly; use one of the subclasses instead.

    The model requires that the system starts out entirely in one of the ground or
    excited states, specified using :meth:`__init__`'s :param:`start_excited` parameter.

    The probability of transition from one state to the other is calculated as
        `P_trans = 1 / 2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]`
    where
        - t is the duration of interaction between qubit and driving field
        - W = sqrt(omega^2 + delta^2)
        - delta is the detuning of the driving field from resonance
        - omega is the Rabi frequency
        - tau is the decay time constant.

    Independent variables:
        - t_pulse: duration of driving pulse including dead time. The duration of the
            interaction is given by t = max(0, t_pulse - t_dead).
        - w: frequency of driving pulse relative to the reference frequency `w_0`, given
            by `delta = w - w_0`

    Model parameters:
        - P_readout_e: excited state readout level
        - P_readout_g: ground state readout level
        - omega: Rabi frequency
        - tau: decay time constant (fixed to infinity by default)
        - t_dead: dead time (fixed to 0 by default)
        - w_0: resonance frequency offset

    Derived parameters:
        - t_pi: Pi-time, calculated as t_pi = pi / omega
        - t_pi_2: Pi/2-time, calculated as t_pi_2 = t_pi / 2
        - f_0: Offset of resonance from zero of frequency variable in linear units

    All frequencies are in angular units.
    """

    def __init__(self, start_excited: bool):
        super().__init__()
        self.start_excited = start_excited

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        # Beware if you're sub-typing this!
        # This is not the standard type for `x`; we rely on the implementation of `func`
        # to change the type of `x` for us (see the RabiFlopFreq / RabiFlopTime
        # implementations)
        x: Tuple[
            Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
        ],
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
        omega: ModelParameter(lower_bound=0.0),
        tau: ModelParameter(lower_bound=0.0, fixed_to=np.inf),
        t_dead: ModelParameter(lower_bound=0.0, fixed_to=0.0),
        w_0: ModelParameter(),
    ) -> Array[("num_samples",), np.float64]:
        """:param x: tuple of (t_pulse, w)"""
        t = np.clip(x[0] - t_dead, a_min=0.0, a_max=None)
        delta = x[1] - w_0
        W = np.sqrt(omega**2 + delta**2)

        P_trans = (
            0.5
            * np.divide(omega**2, W**2, out=np.zeros_like(W), where=(W != 0.0))
            * (1 - np.exp(-t / tau) * np.cos(W * t))
        )
        P_e = 1 - P_trans if self.start_excited else P_trans

        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    # pytype: enable=invalid-annotation

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        omega = fitted_params["omega"]
        t_pi = np.pi / omega
        t_pi_2 = t_pi / 2

        omega_err = fit_uncertainties["omega"]
        t_dead_err = fit_uncertainties["t_dead"]

        derived_params = {}
        derived_params["t_pi"] = t_pi
        derived_params["t_pi_2"] = t_pi_2

        derived_uncertainties = {}
        derived_uncertainties["t_pi"] = np.sqrt(
            t_dead_err**2 + (omega_err * np.pi / (omega**2)) ** 2
        )
        derived_uncertainties["t_pi_2"] = np.sqrt(
            t_dead_err**2 + (omega_err * np.pi / 2 * (omega**2)) ** 2
        )

        if "w_0" in fitted_params:
            derived_params["f_0"] = fitted_params["w_0"] / (2 * np.pi)
            derived_uncertainties["f_0"] = fit_uncertainties["w_0"] / (2 * np.pi)

        return derived_params, derived_uncertainties


class RabiFlopFreq(RabiFlop):
    """Fit model for Rabi flopping pulse detuning scans.

    This model calculates the measurement outcomes for damped Rabi flops when the
    pulse duration is kept fixed and only its frequency is varied. The pulse duration is
    specified using a new `t_pulse` model parameter.
    """

    def __init__(self, start_excited: bool):
        super().__init__(start_excited)

        self.parameters["t_pulse"] = ModelParameter(lower_bound=0.0)

        self.parameters["t_pulse"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Returns measurement outcome as function of pulse frequency.

        :param x: Angular frequency
        """
        param_values = param_values.copy()
        t_pulse = param_values.pop("t_pulse")
        return self._func(
            (t_pulse, x), **param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        model_parameters["t_dead"].heuristic = 0.0
        model_parameters["tau"].heuristic = np.inf

        if self.start_excited:
            model_parameters["P_readout_e"].heuristic = y[0]
            model_parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            model_parameters["P_readout_g"].heuristic = y[0]
            model_parameters["P_readout_e"].heuristic = abs(1 - y[0])

        y0_param = "P_readout_e" if self.start_excited else "P_readout_g"

        # A common use of this model is finding `w_0` when the Rabi frequency and pulse
        # duration are known. In this case we don't need to rely on the Sinc2
        # approximation
        unknowns = set()
        for param, param_data in model_parameters.items():
            try:
                param_data.get_initial_value()
            except ValueError:
                unknowns.add(param)

        if unknowns == {"w_0"}:
            omega, spectrum = get_spectrum(x, y, trim_dc=True)
            w_0 = self.find_x_offset_sym_peak(
                x=x,
                y=y,
                parameters=model_parameters,
                omega=omega,
                spectrum=spectrum,
                omega_cut_off=model_parameters["t_pulse"].get_initial_value(),
                x_offset_param_name="w_0",
                y_offset_param_name=y0_param,
            )
            model_parameters["w_0"].heuristic = w_0
            return

        # There isn't a simple analytic form for the Fourier transform of a Rabi
        # flop in the general case. However in the low pulse area limit (and
        # ignoring decay etc) the Rabi flop function tends to the sinc^2 function:
        #   (omega * t / 2)^2 * sinc^2(delta * t / 2)
        # NB np.sinc(x) = np.sin(pi * x) / (pi * x)
        # This heuristic breaks down when: omega * t_pulse ~ pi
        model = Sinc2()
        y0 = model_parameters[y0_param].get_initial_value()
        model.parameters["y0"].fixed_to = y0
        fit = NormalFitter(x, y, model)

        model_parameters["t_pulse"].heuristic = 2 * fit.values["w"]
        t_pulse = model_parameters["t_pulse"].get_initial_value()
        model_parameters["omega"].heuristic = (
            2 * np.sqrt(np.abs(fit.values["a"])) / t_pulse
        )

        if model_parameters["w_0"].has_user_initial_value():
            return

        # The user hasn't told us what w_0 is so we need to find a heuristic value
        # In addition to going off the Sinc^2, we use a simple sampling-based heuristic
        x_sinc = fit.values["x0"]

        # Test out all points with a contrast of 30% of more. NB the fitter
        # automatically rescales our y-data so this assumes we have one point at
        # sufficiently high contrast for the y-axis rescaling to not do much!
        x_sample = x[np.argwhere(np.abs(y - y0) > 0.3)]
        x_trial = np.append(x_sample, [x_sinc])
        w_0, _ = self.param_min_sqrs(
            x, y, model_parameters, scanned_param="w_0", scanned_param_values=x_trial
        )
        model_parameters["w_0"].heuristic = w_0


class RabiFlopTime(RabiFlop):
    """Fit model for Rabi flopping pulse duration scans.

    This model calculates the measurement outcome for damped Rabi flops when the
    frequency of the pulse is kept fixed and only its duration is varied.

    Since the detuning is not scanned as an independent variable, we replace `w_0` with
    a new model parameter `delta`, defined by: `delta = |w - w_0|`.
    """

    def __init__(self, start_excited: bool):
        super().__init__(start_excited)

        self.parameters["delta"] = ModelParameter()
        del self.parameters["w_0"]

        self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Returns measurement outcome as function of pulse duration.

        :param x: Pulse duration
        """
        param_values = param_values.copy()
        delta = param_values.pop("delta")
        param_values["w_0"] = 0.0
        return self._func((x, delta), **param_values)  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        model_parameters["t_dead"].heuristic = 0.0
        model_parameters["tau"].heuristic = np.inf

        if self.start_excited:
            model_parameters["P_readout_e"].heuristic = y[0]
            model_parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            model_parameters["P_readout_g"].heuristic = y[0]
            model_parameters["P_readout_e"].heuristic = abs(1 - y[0])

        P_readout_e = model_parameters["P_readout_e"].get_initial_value()
        P_readout_g = model_parameters["P_readout_g"].get_initial_value()

        model = Sinusoid()
        if P_readout_e >= P_readout_g:
            model.parameters["phi"].fixed_to = (
                np.pi / 2 if self.start_excited else 3 * np.pi / 2
            )
        else:
            model.parameters["phi"].fixed_to = (
                3 * np.pi / 2 if self.start_excited else np.pi / 2
            )

        fit = NormalFitter(x, y, model)
        W = fit.values["omega"]
        model_parameters["omega"].heuristic = np.sqrt(2 * fit.values["a"]) * W
        omega = model_parameters["omega"].get_initial_value()

        if W >= omega:
            model_parameters["delta"].heuristic = np.sqrt(W**2 - omega**2)
        else:
            # can't use param_min_sqrs because omega and delta are coupled
            deltas = np.linspace(0, W / 2, 10)
            omegas = np.sqrt(W**2 - np.power(deltas, 2))
            costs = np.zeros_like(deltas)

            initial_values = {
                param: param_data.get_initial_value()
                for param, param_data in model_parameters.items()
                if param != "delta"
            }

            for idx in range(len(deltas)):
                initial_values["delta"] = deltas[idx]
                initial_values["omega"] = omegas[idx]
                y_idx = self.func(x, initial_values)
                costs[idx] = np.sqrt(np.sum(np.power(y - y_idx, 2)))
            opt_idx = np.argmin(costs)

            model_parameters["delta"].heuristic = deltas[opt_idx]
            model_parameters["omega"].heuristic = omegas[opt_idx]
