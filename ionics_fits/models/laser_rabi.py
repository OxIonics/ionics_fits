import copy
import inspect
from typing import Callable, Dict, Tuple, TYPE_CHECKING

import numpy as np
from scipy import special

from .quantum_phys import coherent_state_probs, thermal_state_probs
from . import rabi
from .. import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_fock_states = float


class LaserFlop(Model):
    """
    Base class for damped Rabi flops with finite Lamb-Dicke parameter.

    This model calculates the measurement probability for a two-state system undergoing
    damped Rabi flopping to be measured to be in the excited state, given by:
        P = P_readout_g + (P_readout_e - P_readout_g) * P_e
    where P_e is the time-dependent population in the excited state, while
    P_readout_g and P_readout_e denote the individual readout levels. The ground and
    excited states are denoted by |g> and |e> respectively.

    The model requires that the system starts out in either |g> or |e>,
    specified by passing :param:`start_excited` to :meth:`__init__`.

    This class does not support fitting directly, use one of the subclasses instead.

    Independent variables:
        - t_pulse: Duration of driving pulse including dead time. The true duration of
            interaction is calculated as t = max(0, t_pulse - t_dead).
        - w: Variable that determines frequency of driving pulse. This does not have to
            be the absolute frequency, but may instead be measured relative to some
            arbitrary reference frequency. The detuning from resonance is calculated
            as delta = w - w_0.

    Model parameters:
        - P_readout_e: Readout level for state |e>
        - P_readout_g: Readout level for state |g>
        - eta: Lamb-Dicke parameter
        - omega: carrier Rabi frequency
        - tau: Decay time constant (fixed to infinity by default)
        - t_dead: Dead time (fixed to 0 by default)
        - w_0: Offset of resonance from zero of frequency variable

    The model additionally gains any parameters associated with the specified Fock state
    distribution function.

    Derived parameters:
        - t_pi: Pi-time, calculated as t_pi = pi / omega
        - t_pi_2: Pi/2-time, calculated as t_pi_2 = t_pi / 2
        - f_0: Offset of resonance from zero of frequency variable in linear units

    All frequencies are in angular units.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        distribution_fun: Callable[..., Array[("num_fock_states",), np.float64]],
        n_max: int = 15,
    ):
        """
        :param start_excited: if True the qubit starts in the excited state
        :param sideband_index: change in motional state due to a pi-pulse starting from
          the spin ground-state.
        :param distribution_fun: function returning an array of Fock state occupation
          probabilities. The distribution function's first argument should be the
          maximum Fock state to include in the simulation (the returned array has n_max
          + 1 elements). Subsequent arguments should be `ModelParameter`s used to
          parametrise the distribution.
        :param n_max: maximum Fock state used in the simulation
        """
        super().__init__()
        self.distribution_fun = distribution_fun
        self.start_excited = start_excited
        self.sideband_index = sideband_index
        self.n_max = n_max

        spec = inspect.getfullargspec(distribution_fun)
        self.distribution_params = {
            name: copy.deepcopy(spec.annotations[name]) for name in spec.args[1:]
        }

        assert all(
            [
                isinstance(param, ModelParameter)
                for param in self.distribution_params.values()
            ]
        ), "Distribution parameters must be instances of `ModelParameter`"
        assert not any(
            [
                param_name in self.parameters.keys()
                for param_name in self.distribution_params.keys()
            ]
        ), "Distribution parameter names must not match model parameter names"
        self.parameters.update(self.distribution_params)

        n = np.arange(self.n_max + 1)
        self._n_min = np.minimum(n, n + self.sideband_index)
        self._n_max = np.maximum(n, n + self.sideband_index)

        # sqrt(n_min!/n_max!)
        self.fact = np.exp(
            0.5 * (special.gammaln(self._n_min + 1) - special.gammaln(self._n_max + 1))
        )
        self.fact[self._n_min < 0] = 0

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
        eta: ModelParameter(lower_bound=0.0),
        omega: ModelParameter(lower_bound=0.0),
        tau: ModelParameter(lower_bound=0.0, fixed_to=np.inf),
        t_dead: ModelParameter(lower_bound=0.0, fixed_to=0.0),
        w_0: ModelParameter(),
        **kwargs,  # Fock state distribution function parameters
    ) -> Array[("num_samples",), np.float64]:
        """
        Return measurement probability.

        :param x: Tuple (t_pulse, w) of ndarrays containing pulse duration and
            angular frequency of driving field. They must have shapes such
            that they are broadcastable.
        """
        # coupling between |g>|n> and |e>|n+sideband_index>
        omega_vec = (
            omega
            * np.exp(-0.5 * np.power(eta, 2))
            * np.power(eta, np.abs(self.sideband_index))
            * self.fact
            * special.eval_genlaguerre(
                self._n_min, np.abs(self.sideband_index), np.power(eta, 2)
            )
        )

        t_vec = np.clip(x[0] - t_dead, a_min=0, a_max=None)
        detuning_vec = x[1]
        t, detuning, omega = np.meshgrid(t_vec, detuning_vec, omega_vec, indexing="ij")

        # P_transition = 1 / 2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]
        W = np.sqrt(np.power(omega, 2) + np.power(detuning, 2))

        P_trans = (
            0.5
            * np.power(np.divide(omega, W, out=np.zeros_like(W), where=(W != 0.0)), 2)
            * (1 - np.exp(-t / tau) * np.cos(W * t))
        )

        initial_state = self.distribution_fun(self.n_max, **kwargs)
        P_i = np.tile(initial_state, (omega.shape[0], omega.shape[1], 1))
        P_e = np.sum(P_i * P_trans, axis=-1).squeeze()

        P_e = 1 - P_e if self.start_excited else P_e

        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    # pytype: enable=invalid-annotation

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
        return rabi.RabiFlop.calculate_derived_params(
            self, x, y, fitted_params, fit_uncertainties
        )


class LaserFlopFreq(LaserFlop):
    """
    Fit model for Rabi flopping pulse detuning scans.

    This model calculates the measurement outcome for damped Rabi flops when the
    pulse duration is kept fixed and only its frequency is varied. The pulse duration is
    specified using a new `t_pulse` model parameter.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        distribution_fun: Callable[..., Array[("num_fock_states",), np.float64]],
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=distribution_fun,
            n_max=n_max,
        )

        self.parameters["t_pulse"] = ModelParameter(lower_bound=0.0)

        self.parameters["t_pulse"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """
        Return measurement probability as function of pulse frequency.

        :param x: Angular frequency
        """
        param_values = param_values.copy()
        t_pulse = param_values.pop("t_pulse")
        return super()._func(
            (t_pulse, x), **param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        # Pick sensible starting values which are usually good enough for the fit to
        # converge from.
        model_parameters["eta"].heuristic = 0.1
        if "n_bar" in model_parameters.keys():
            model_parameters["n_bar"] = 1
        if "alpha" in model_parameters.keys():
            model_parameters["alpha"] = 1

        rabi.RabiFlopFreq.estimate_parameters(
            self, x=x, y=y, model_parameters=model_parameters
        )


class LaserFlopTime(LaserFlop):
    """
    Fit model for laser flopping pulse duration scans.

    This model calculates the measurement outcome for damped Rabi flops when the
    frequency of the pulse is kept fixed and only its duration is varied.

    Since the detuning is not scanned as an independent variable, we replace `w_0` with
    a new model parameter `delta`, defined by: delta = |w - w_0|.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        distribution_fun: Callable[..., Array[("num_fock_states",), np.float64]],
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=distribution_fun,
            n_max=n_max,
        )
        self.parameters["delta"] = ModelParameter()
        del self.parameters["w_0"]

        self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """
        Return measurement probability as function of pulse duration.

        :param x: Pulse duration
        """
        param_values = param_values.copy()
        delta = param_values.pop("delta")
        param_values["w_0"] = 0.0
        return super()._func(
            (x, delta), **param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        # Pick sensible starting values which are usually good enough for the fit to
        # converge from.
        model_parameters["eta"].heuristic = 0.1
        if "n_bar" in model_parameters.keys():
            model_parameters["n_bar"] = 1
        if "alpha" in model_parameters.keys():
            model_parameters["alpha"] = 1

        rabi.RabiFlopTime.estimate_parameters(
            self, x=x, y=y, model_parameters=model_parameters
        )


class LaserFlopTimeThermal(LaserFlopTime):
    """Laser flopping pulse duration scans starting from a thermal state"""

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=thermal_state_probs,
            n_max=n_max,
        )


class LaserFlopTimeCoherent(LaserFlopTime):
    """Laser flopping pulse duration scans starting from a coherent state"""

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=coherent_state_probs,
            n_max=n_max,
        )


class LaserFlopFreqThermal(LaserFlopFreq):
    """Laser flopping frequency scans starting from a thermal state"""

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=thermal_state_probs,
            n_max=n_max,
        )


class LaserFlopFreqCoherent(LaserFlopFreq):
    """Laser flopping frequency scans starting from a coherent state"""

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 15,
    ):
        super().__init__(
            start_excited=start_excited,
            sideband_index=sideband_index,
            distribution_fun=coherent_state_probs,
            n_max=n_max,
        )
