from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..normal import NormalFitter
from ..utils import scale_undefined, scale_x, scale_x_inv, scale_y
from . import heuristics
from .heuristics import get_spectrum
from .sinc import Sinc2
from .sinusoid import Sinusoid


class RabiFlop(Model):
    r"""Base class for damped Rabi flopping.

    This model calculates measurement outcomes for two-state systems undergoing damped
    Rabi oscillations, defined by::

        P = P_readout_g + (P_readout_e - P_readout_g) * P_e

    where ``P_e`` is the (time-dependent) population in the excited state and
    ``P_readout_g`` and ``P_readout_e`` are the readout levels (measurement outcomes
    when the qubit is in one state).

    This class does not support fitting directly; use one of the subclasses instead.

    The model requires that the system starts out entirely in one of the ground or
    excited states, specified using :meth:`__init__`\'s
    ``start_excited`` parameter.

    The probability of transition from one state to the other is calculated as::

        P_trans = 1 / 2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]

    where:
        * ``t`` is the duration of interaction between qubit and driving field
        * ``W = sqrt(omega^2 + delta^2)``
        * ``delta`` is the detuning of the driving field from resonance
        * ``omega`` is the Rabi frequency
        * ``tau`` is the decay time constant.

    Independent variables:
        * ``t_pulse``: duration of driving pulse including dead time. The duration of
          the interaction is given by ``t = max(0, t_pulse - t_dead)``.
        * ``w``: frequency of driving pulse relative to the reference frequency ``w_0``,
          given by ``delta = w - w_0``

    All frequencies are in angular units.
    """

    def __init__(self, start_excited: bool):
        """
        :param start_excited: if ``True`` the system is assumed to start in the excited
            state, other wise it is assumed to start in the ground state.
        """
        super().__init__()
        self.start_excited = start_excited

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [False]

    def func(self, x: Tuple[TX, TX], param_values: Dict[str, float]) -> TY:
        """Evaluates the model at a given set of pulse durations and frequencies and
        with a given set of parameter values and returns the result.

        To use the model as a function outside of a fit,
        :meth:`~ionics_fits.common.Model.__call__` generally
        provides a more convenient interface.

        :param x: Tuple of ``(t_pulse, w)``
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        return self._func(x, **param_values)

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: Tuple[TX, TX],
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
        omega: ModelParameter(lower_bound=0.0, scale_func=scale_undefined),
        tau: ModelParameter(
            lower_bound=0.0, fixed_to=np.inf, scale_func=scale_undefined
        ),
        t_dead: ModelParameter(
            lower_bound=0.0, fixed_to=0.0, scale_func=scale_undefined
        ),
        w_0: ModelParameter(scale_func=scale_undefined),
    ) -> TY:
        """Return measurement probability.

        :param x: tuple of ``(t_pulse, w)``. Subclasses should override func to
            map this onto the appropriate input data.
        :param P_readout_e: excited state readout level
        :param P_readout_g: ground state readout level
        :param omega: Rabi frequency
        :param tau: decay time constant (fixed to infinity by default)
        :param t_dead: dead time (fixed to 0 by default)
        :param w_0: resonance frequency offset
        """
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

    # pytype: enable=invalid-annotation,signature-mismatch

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Derived parameters:
            - ``t_pi``: Pi-time, calculated as ``t_pi = pi / omega``
            - ``t_pi_2``: Pi/2-time, calculated as ``t_pi_2 = t_pi / 2``
            - ``f_0``: Offset of resonance from zero of frequency variable in linear
              units
        """
        omega = fitted_params["omega"]
        t_pi = np.pi / omega
        t_pi_2 = t_pi / 2

        omega_err = fit_uncertainties["omega"]

        derived_params = {}
        derived_params["t_pi"] = t_pi
        derived_params["t_pi_2"] = t_pi_2

        derived_uncertainties = {}
        derived_uncertainties["t_pi"] = omega_err * np.pi / (omega**2)
        derived_uncertainties["t_pi_2"] = derived_uncertainties["t_pi"] / 2

        if "w_0" in fitted_params:
            derived_params["f_0"] = fitted_params["w_0"] / (2 * np.pi)
            derived_uncertainties["f_0"] = fit_uncertainties["w_0"] / (2 * np.pi)

        return derived_params, derived_uncertainties


class RabiFlopFreq(RabiFlop):
    """Fit model for Rabi flopping frequency scans.

    This model calculates the measurement outcomes for damped Rabi flops when the
    pulse duration is kept fixed and only its frequency is varied. The pulse duration is
    specified using a new ``t_pulse`` model parameter.

    See also :class:`~ionics_fits.models.rabi.RabiFlop`.
    """

    def __init__(self, start_excited: bool):
        super().__init__(start_excited)

        self.parameters["t_pulse"] = ModelParameter(
            lower_bound=0.0, scale_func=scale_x_inv()
        )

        self.parameters["omega"].scale_func = scale_x()
        self.parameters["w_0"].scale_func = scale_x()

        self.parameters["tau"].scale_func = scale_x_inv()
        self.parameters["t_dead"].scale_func = scale_x_inv()

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        param_values = param_values.copy()
        t_pulse = param_values.pop("t_pulse")
        return self._func((t_pulse, x), **param_values)

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        self.parameters["t_dead"].heuristic = 0.0

        if self.start_excited:
            self.parameters["P_readout_e"].heuristic = y[0]
            self.parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            self.parameters["P_readout_g"].heuristic = y[0]
            self.parameters["P_readout_e"].heuristic = abs(1 - y[0])

        y0_param = "P_readout_e" if self.start_excited else "P_readout_g"

        # A common use of this model is finding `w_0` when the Rabi frequency and pulse
        # duration are known. In this case we don't need to rely on the Sinc2
        # approximation
        unknowns = set()
        for param, param_data in self.parameters.items():
            try:
                param_data.get_initial_value()
            except ValueError:
                unknowns.add(param)

        if unknowns == {"w_0"}:
            omega, spectrum = get_spectrum(x, y, trim_dc=True)
            w_0 = heuristics.find_x_offset_sym_peak_fft(
                model=self,
                x=x,
                y=y,
                omega=omega,
                spectrum=spectrum,
                omega_cut_off=self.parameters["t_pulse"].get_initial_value(),
                x_offset_param_name="w_0",
                y_offset_param_name=y0_param,
            )
            self.parameters["w_0"].heuristic = w_0
            return

        # There isn't a simple analytic form for the Fourier transform of a Rabi
        # flop in the general case. However in the low pulse area limit (and
        # ignoring decay etc) the Rabi flop function tends to the sinc^2 function:
        #   (omega * t / 2)^2 * sinc^2(delta * t / 2)
        # NB np.sinc(x) = np.sin(pi * x) / (pi * x)
        # This heuristic breaks down when: omega * t_pulse ~ pi
        model = Sinc2()
        y0 = self.parameters[y0_param].get_initial_value()
        model.parameters["y0"].fixed_to = y0
        fit = NormalFitter(x, y, model)

        self.parameters["t_pulse"].heuristic = 2 * fit.values["w"]
        t_pulse = self.parameters["t_pulse"].get_initial_value()
        self.parameters["omega"].heuristic = (
            2 * np.sqrt(np.abs(fit.values["a"])) / t_pulse
        )

        if self.parameters["w_0"].has_user_initial_value():
            return

        # The user hasn't told us what w_0 is so we need to find a heuristic value
        # In addition to going off the Sinc^2, we use a simple sampling-based heuristic
        x_sinc = fit.values["x0"]

        # Test out all points with a contrast of 30% of more. NB the fitter
        # automatically rescales our y-data so this assumes we have one point at
        # sufficiently high contrast for the y-axis rescaling to not do much!
        x_sample = x[np.argwhere(np.abs(y - y0) > 0.3)]
        x_trial = np.append(x_sample, [x_sinc])
        w_0, _ = heuristics.param_min_sqrs(
            model=self, x=x, y=y, scanned_param="w_0", scanned_param_values=x_trial
        )
        self.parameters["w_0"].heuristic = w_0

        self.parameters["tau"].heuristic = 10 * t_pulse


class RabiFlopTime(RabiFlop):
    """Fit model for Rabi flopping pulse duration scans.

    This model calculates the measurement outcome for damped Rabi flops when the
    frequency of the pulse is kept fixed and only its duration is varied.

    Since the detuning is not scanned as an independent variable, we replace ``w_0``
    with a new model parameter ``delta``, defined by: ``delta = |w - w_0|``.

    See also :class:`~ionics_fits.models.rabi.RabiFlop`.
    """

    def __init__(self, start_excited: bool):
        super().__init__(start_excited)

        self.parameters["delta"] = ModelParameter(scale_func=scale_x_inv())
        del self.parameters["w_0"]

        self.parameters["omega"].scale_func = scale_x_inv()

        self.parameters["tau"].scale_func = scale_x()
        self.parameters["t_dead"].scale_func = scale_x()

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        param_values = param_values.copy()
        delta = param_values.pop("delta")
        param_values["w_0"] = 0.0
        return self._func((x, delta), **param_values)

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        self.parameters["t_dead"].heuristic = 0.0
        self.parameters["tau"].heuristic = 10 * np.ptp(x)

        if self.start_excited:
            self.parameters["P_readout_e"].heuristic = y[0]
            self.parameters["P_readout_g"].heuristic = abs(1 - y[0])
        else:
            self.parameters["P_readout_g"].heuristic = y[0]
            self.parameters["P_readout_e"].heuristic = abs(1 - y[0])

        P_readout_e = self.parameters["P_readout_e"].get_initial_value()
        P_readout_g = self.parameters["P_readout_g"].get_initial_value()

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
        self.parameters["omega"].heuristic = np.sqrt(2 * fit.values["a"]) * W
        omega = self.parameters["omega"].get_initial_value()

        if W >= omega:
            self.parameters["delta"].heuristic = np.sqrt(W**2 - omega**2)
        else:
            # can't use param_min_sqrs because omega and delta are coupled
            deltas = np.linspace(0, W / 2, 10)
            omegas = np.sqrt(W**2 - deltas**2)
            costs = np.zeros_like(deltas)

            initial_values = {
                param: param_data.get_initial_value()
                for param, param_data in self.parameters.items()
                if param != "delta"
            }

            for idx in range(len(deltas)):
                initial_values["delta"] = deltas[idx]
                initial_values["omega"] = omegas[idx]
                y_idx = self.func(x, initial_values)
                costs[idx] = np.sqrt(np.sum((y - y_idx) ** 2))
            opt_idx = np.argmin(costs)

            self.parameters["delta"].heuristic = deltas[opt_idx]
            self.parameters["omega"].heuristic = omegas[opt_idx]

        # Corner-case: if the time axis starts from t_0 >> t_pi the above heuristic
        # can fail. This is because the accuracy of the sinusoid fit is limited by the
        # step size in frequency space. Once the uncertainty in the Rabi frequency
        # estimate becomes such that we don't know if we've done n or (n + 1) flops
        # before t_0 the fits will start failing.
        d_omega = 2 * np.pi / np.ptp(x)  # approx uncertainty in Rabi freq from FFT
        t_pi = np.pi / self.parameters["omega"].get_initial_value()
        d_t_pi = np.pi / d_omega
        n_pi = min(x) / t_pi
        err = d_t_pi * n_pi  # number of t_pi worth of uncertainty in the fit

        if err > 0.1:
            n_pi_min = max(n_pi - 4 * err, 0.0)
            n_pi_max = n_pi + 4 * err
            num_pts = int(np.rint(20 * (n_pi_max - n_pi_min)))

            omega_min = np.pi * n_pi_min / min(x)
            omega_max = np.pi * n_pi_max / min(x)
            omegas = np.linspace(omega_min, omega_max, num_pts)

            self.parameters["omega"].heuristic, _ = heuristics.param_min_sqrs(
                model=self,
                x=x,
                y=y,
                scanned_param="omega",
                scanned_param_values=omegas,
            )
