from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model, ModelParameter
from ..utils import Array


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

        if num_qubits not in [1, 2, 3]:
            raise ValueError("Model only supports 1 or 2 qubits")
        if walsh_idx not in [0, 1, 2, 3]:
            raise ValueError("Unsupported Walsh index")

        self.num_qubits = num_qubits
        self.start_excited = start_excited

        if walsh_idx == 0:
            self.segment_durations = [1]
        elif walsh_idx == 1:
            self.segment_durations = [1, 1]
        elif walsh_idx == 2:
            self.segment_durations = [1, 1, 1, 1]
        else:
            self.segment_durations = [1, 2, 1]
        self.segment_durations = np.array(self.segment_durations, dtype=float)
        self.segment_durations /= sum(self.segment_durations)

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
        for duration in self.segment_durations:
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
        for outer_idx, outer_duration in enumerate(self.segment_durations):
            t_outer_f = t_outer_i + outer_duration * t_pulse

            inner_segments = self.segment_durations[0:outer_idx]
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
        x: Tuple[Array[("num_samples",), np.float64]],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_params["f_0"] = fitted_params["w_0"] / (2 * np.pi)

        derived_uncertainties = {}
        derived_uncertainties["f_0"] = fit_uncertainties["w_0"] / (2 * np.pi)

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

        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale

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
        # These heuristics are pretty basic, but seem to work fine...
        model_parameters["n_bar"].heuristic = 0.0
        model_parameters["delta"].heuristic = 0.0
        model_parameters["omega"].heuristic = np.pi / max(x)


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

        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["t_pulse"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale

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
        # These heuristics are pretty basic but seem to work fine...
        model_parameters["n_bar"].heuristic = 0.0
        model_parameters["w_0"].heuristic = 0.0

        if model_parameters["t_pulse"].has_user_initial_value():
            t_pulse = model_parameters["t_pulse"].get_initial_value()
            model_parameters["omega"].heuristic = np.pi / t_pulse
        else:
            model_parameters["omega"].heuristic = max(abs(x))

        omega = model_parameters["omega"].get_initial_value()
        model_parameters["t_pulse"].heuristic = np.pi / omega
