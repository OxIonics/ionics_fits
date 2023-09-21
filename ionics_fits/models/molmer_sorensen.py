from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model, ModelParameter
from ..utils import Array
from . import heuristics


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class MolmerSorensen(Model):
    """Base class for Mølmer–Sørensen interactions.

    This model calculates the time-dependent populations for one or two qubits
    coupled to a single motional mode undergoing a Mølmer–Sørensen type
    interaction.

    It requires that the initial spin states of all qubits are the same
    and either |g> or |e> - different initial states for each qubit or initial
    states which are superpositions of spin eigenstates are are not supported.

    For single-qubit interactions, the model has one y channel, giving the
    excited-state population at the end of the interaction.

    For two-qubit interactions, the model has three y channels - P_gg, P_1e,
    P_ee - giving the probabilities of 0, 1 or 2 ions being in the excited state
    at the end of the interaction duration.

    Modulation of the sign of the spin-dependent force according to a Walsh
    function is supported.

    The motion's initial state must be a thermal distribution.

    This class does not support fitting directly; use one of its subclasses
    instead.

    Independent variables:
        - t_pulse: total interaction duration.
        - w: detuning of red/blue sideband tones relative to reference frequency
            `w_0`. The interaction detuning is given by `delta = w - w_0`.

    Model parameters:
        - omega: sideband Rabi frequency
        - w_0: resonance frequency offset
        - n_bar: average initial occupancy of the motional mode (fixed to 0 by
            default)

    All frequencies are in angular units.
    """

    def __init__(self, num_qubits: int, start_excited: bool, walsh_idx: int):
        """
        :param num_qubits: number of qubits (must be 1 or 2)
        :param walsh_idx: Index of Walsh function
        :param start_excited: If True, all qubits start in |e>, otherwise they
            start in |g>
        """
        super().__init__()

        if num_qubits not in [1, 2]:
            raise ValueError("Model only supports 1 or 2 qubits")
        if walsh_idx not in [0, 1, 2, 3]:
            raise ValueError("Unsupported Walsh index")

        self.num_qubits = num_qubits
        self.start_excited = start_excited
        self.walsh_idx = walsh_idx

    def get_num_y_channels(self) -> int:
        return [1, 3][self.num_qubits - 1]

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Tuple[
            Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
        ],
        omega: ModelParameter(lower_bound=0.0),
        w_0: ModelParameter(),
        n_bar: ModelParameter(
            lower_bound=0.0, fixed_to=0.0, scale_func=lambda x_scale, y_scale, _: 1
        ),
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:

        t_pulse = x[0]
        delta = x[1] - w_0
        data_shape = np.broadcast(t_pulse, delta).shape

        if self.walsh_idx == 0:
            segment_durations = [1]
        elif self.walsh_idx == 1:
            segment_durations = [1, 1]
        elif self.walsh_idx == 2:
            segment_durations = [1, 1, 1, 1]
        else:
            segment_durations = [1, 2, 1]
        segment_durations = np.array(segment_durations, dtype=float)
        segment_durations /= sum(segment_durations)

        def displacement(delta, t_i, t_f):
            alpha = np.full(shape=data_shape, fill_value=-1j * t_pulse)
            alpha = (
                0.5
                * omega
                * np.divide(
                    np.exp(-1j * delta * t_f) - np.exp(-1j * delta * t_i),
                    delta,
                    out=alpha,
                    where=(delta != 0),
                )
            )
            return alpha

        alpha = np.zeros(data_shape, dtype=np.complex128)
        t_i = 0
        sign = +1
        for duration in segment_durations:
            t_f = t_i + duration * t_pulse
            alpha += sign * displacement(delta=delta, t_i=t_i, t_f=t_f)
            t_i = t_f
            sign *= -1

        gamma = np.abs(alpha) ** 2 * (n_bar + 1 / 2)

        if self.num_qubits == 1:
            P = 0.5 * (1 + np.exp(-4 * gamma))
            if self.start_excited:
                return P
            else:
                return 1 - P

        # piecewise integration of the geometric phase
        phi = delta * t_pulse
        t_outer_i = 0
        for outer_idx, outer_duration in enumerate(segment_durations):
            t_outer_f = t_outer_i + outer_duration * t_pulse

            inner_segments = segment_durations[0:outer_idx]
            t_inner_i = 0
            for inner_idx, inner_duration in enumerate(inner_segments):
                t_inner_f = t_inner_i + inner_duration * t_pulse
                sign = (-1) ** (outer_idx - inner_idx)
                phi += sign * (
                    np.sin(delta * (t_outer_f - t_inner_f))
                    - np.sin(delta * (t_outer_f - t_inner_i))
                    - np.sin(delta * (t_outer_i - t_inner_f))
                    + np.sin(delta * (t_outer_i - t_inner_i))
                )
                t_inner_i = t_inner_f

            phi -= np.sin(delta * (t_outer_f - t_inner_i))
            t_outer_i = t_outer_f

        phi *= np.square(
            np.divide(omega, 2 * delta, out=np.zeros(data_shape), where=(delta != 0))
        )

        # Calculate spin transition probabilities
        P_0 = (
            1 / 8 * (3 + np.exp(-16 * gamma) + 4 * np.cos(4 * phi) * np.exp(-4 * gamma))
        )

        P_1 = 1 / 4 * (1 - np.exp(-16 * gamma))

        P_2 = (
            1 / 8 * (3 + np.exp(-16 * gamma) - 4 * np.cos(4 * phi) * np.exp(-4 * gamma))
        )

        if self.start_excited:
            P_gg = P_2
            P_ee = P_0
        else:
            P_gg = P_0
            P_ee = P_2

        return np.vstack([P_gg, P_1, P_ee])

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        w_0_param = "w_0" if "w_0" in fitted_params.keys() else "delta"
        derived_params["f_0"] = fitted_params[w_0_param] / (2 * np.pi)

        derived_uncertainties = {}
        derived_uncertainties["f_0"] = fit_uncertainties[w_0_param] / (2 * np.pi)

        return derived_params, derived_uncertainties

    # pytype: enable=invalid-annotation


class MolmerSorensenTime(MolmerSorensen):
    """
    Fit model for Mølmer–Sørensen pulse duration scans.

    This model calculates the populations for Mølmer–Sørensen interactions
    when the interaction detuning is kept fixed and only the pulse duration
    is varied.

    Since the detuning is not scanned as an independent variable, we replace
    `w_0` with a new model parameter `delta`, defined by `delta = |w - w_0|`.
    """

    def __init__(self, num_qubits: int, start_excited: bool, walsh_idx: int):
        super().__init__(
            num_qubits=num_qubits, start_excited=start_excited, walsh_idx=walsh_idx
        )

        self.parameters["delta"] = ModelParameter()
        del self.parameters["w_0"]

        # https://github.com/OxIonics/ionics_fits/issues/105
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: None
        self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: None
        self.parameters["n_bar"].scale_func = lambda x_scale, y_scale, _: None

        # self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        # self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        """
        Return populations as function of pulse duration for fixed detuning.

        :param x: pulse duration
        """
        param_values = param_values.copy()
        delta = param_values.pop("delta")
        param_values["w_0"] = 0.0
        return self._func((x, delta), **param_values)  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        model_parameters["n_bar"].heuristic = 0.0
        if (
            not model_parameters["delta"].has_user_initial_value()
            and not model_parameters["omega"].has_user_initial_value()
        ):
            # This case is pretty miserable because there is a very high degree
            # of covariance between delta and omega so even if the heuristics are
            # highly accurate, the optimizer will tend to struggle to converge on
            # the right parameter values.
            model_parameters["delta"].heuristic = 0.0

        if not model_parameters["omega"].has_user_initial_value():
            omegas = np.array([np.linspace(0.1, 10, 25)]) * np.pi / max(x)
            model_parameters["omega"].heuristic, _ = self.param_min_sqrs(
                x=x,
                y=y,
                parameters=model_parameters,
                scanned_param="omega",
                scanned_param_values=omegas,
            )

        elif not model_parameters["delta"].has_user_initial_value():
            omega = model_parameters["omega"].get_initial_value()
            deltas = np.array([np.linspace(0, 10, 25)]) * omega
            model_parameters["delta"].heuristic, _ = self.param_min_sqrs(
                x=x,
                y=y,
                parameters=model_parameters,
                scanned_param="delta",
                scanned_param_values=deltas,
            )


class MolmerSorensenFreq(MolmerSorensen):
    """
    Fit model for Mølmer–Sørensen detuning scans.

    This model calculates the populations for Mølmer–Sørensen interactions
    when the gate duration is kept fixed and only the interaction detuning is
    varied. The pulse duration is specified using a new `t_pulse` model
    parameter.
    """

    def __init__(self, num_qubits: int, start_excited: bool, walsh_idx: int):
        super().__init__(
            num_qubits=num_qubits, start_excited=start_excited, walsh_idx=walsh_idx
        )

        self.parameters["t_pulse"] = ModelParameter(lower_bound=0.0)

        # https://github.com/OxIonics/ionics_fits/issues/105
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: None
        self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: None
        self.parameters["t_pulse"].scale_func = lambda x_scale, y_scale, _: None
        self.parameters["n_bar"].scale_func = lambda x_scale, y_scale, _: None

        # self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: x_scale
        # self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: x_scale
        # self.parameters["t_pulse"].scale_func =
        #    lambda x_scale, y_scale, _: 1 / x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        """
        Return populations as function of detuning for fixed duration.

        :param x: Interaction detuning
        """
        param_values = param_values.copy()
        t_pulse = param_values.pop("t_pulse")
        return self._func(
            (t_pulse, x), **param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        model_parameters["n_bar"].heuristic = 0.0

        # estimate the centre frequency by looking for the symmetry point.
        # Consider only the 1-ion transition probability
        y_test = y if self.num_qubits == 1 else np.atleast_2d(y[1, :])
        w_0 = model_parameters["w_0"].heuristic = heuristics.get_sym_x(x, y_test)

        if (
            model_parameters["t_pulse"].has_user_initial_value()
            and model_parameters["omega"].has_user_initial_value()
        ):
            if not model_parameters["w_0"].has_user_initial_value():
                # The symmetry heuristic is pretty good, but can struggle when
                # the centre frequency is close to the edge of the scan range.
                # Since w_0 is our only unknown here, we throw in a sampling
                # heuristic for good measure.
                w_0_grid, w_0_grid_cost = self.param_min_sqrs(
                    x=x,
                    y=y,
                    parameters=model_parameters,
                    scanned_param="w_0",
                    scanned_param_values=np.linspace(min(x), max(x), 50),
                )
                param_values = {
                    name: param.get_initial_value()
                    for name, param in model_parameters.items()
                }
                y_sym = self.func(x, param_values)
                w_0_sym_cost = np.sqrt(np.sum(np.square(y - y_sym)))
                w_0 = w_0 if w_0_sym_cost < w_0_grid_cost else w_0_grid
                model_parameters["w_0"].heuristic = w_0
            return

        if model_parameters["t_pulse"].has_user_initial_value():
            t_pulse = model_parameters["t_pulse"].get_initial_value()
            model_parameters["omega"].heuristic, _ = self.param_min_sqrs(
                x=x,
                y=y,
                parameters=model_parameters,
                scanned_param="omega",
                scanned_param_values=np.array([np.linspace(0.25, 5, 10)])
                * np.pi
                / t_pulse,
            )
        elif model_parameters["omega"].has_user_initial_value():
            omega = model_parameters["omega"].get_initial_value()
            model_parameters["t_pulse"].heuristic, _ = self.param_min_sqrs(
                x=x,
                y=y,
                parameters=model_parameters,
                scanned_param="t_pulse",
                scanned_param_values=np.array([np.linspace(0.25, 5, 10)])
                * np.pi
                / omega,
            )
        else:
            # this is a bit of a corner case, since the user can usually give us
            # an estimate of one of omega or t_pulse. In the absence of that we
            # fall back to a 2D grid search.
            omegas = np.arange(start=1, stop=25) / 100 * max(abs(x))
            costs = np.zeros_like(omegas)
            t_pulses = np.zeros_like(omegas)
            for idx, omega in np.ndenumerate(omegas):
                model_parameters["omega"].heuristic = omega
                t_pulses[idx], costs[idx] = self.param_min_sqrs(
                    x=x,
                    y=y,
                    parameters=model_parameters,
                    scanned_param="t_pulse",
                    scanned_param_values=np.array([np.linspace(0.1, 5, 20)])
                    * np.pi
                    / omega,
                )
            best = np.argmin(costs)
            model_parameters["omega"].heuristic = omegas[best]
            model_parameters["t_pulse"].heuristic = t_pulses[best]
