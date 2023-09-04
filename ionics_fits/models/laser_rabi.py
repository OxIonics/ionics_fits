import copy
import inspect
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
from scipy import special

from .quantum_phys import (
    coherent_state_probs,
    squeezed_state_probs,
    thermal_state_probs,
)
from . import rabi
from .. import ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_fock_states = float


def make_laser_flop(base_class, distribution_fun):
    """
    Generate subclass for laser Rabi flopping models.

    :param base_class: base class to inherit from
    :param distribution_fun: function returning an array of Fock state occupation
        probabilities. The distribution function's first argument should be the maximum
        Fock state to include in the simulation (the returned array has n_max + 1
        elements). Subsequent arguments should be `ModelParameter`s used to parametrise
        the distribution.
    """

    class LaserFlop(base_class):
        """Base class for damped Rabi flopping with finite Lamb-Dicke parameter.

        This model calculates measurement outcomes for systems containing two
        internal states and moving in a 1D harmonic potential that undergo
        damped Rabi oscillations, defined by:
            `P = P_readout_g + (P_readout_e - P_readout_g) * P_e`
        where `P_e` is the (time-dependent) population in the excited state and
        `P_readout_g` and `P_readout_e` are the readout levels (measurement outcomes
        when the qubit is in one state).

        This class does not support fitting directly; use one of the subclasses instead.

        The model requires that the internal part of the system starts out
        entirely in one of the ground or excited states, specified using
        :meth:`__init__`'s :param:`start_excited` parameter. It further assumes
        that the motional part of the system starts out in a distribution
        over different Fock states, described by the parameter `distribution_fun`.

        Independent variables:
            - t_pulse: duration of driving pulse including dead time. The duration of
                the interaction is given by t = max(0, t_pulse - t_dead).
            - w: frequency of driving pulse relative to the reference frequency `w_0`,
                given by `delta = w - w_0`

        Model parameters:
            - P_readout_e: excited state readout level
            - P_readout_g: ground state readout level
            - eta: Lamb-Dicke parameter
            - omega: carrier Rabi frequency
            - tau: decay time constant (fixed to infinity by default)
            - t_dead: dead time (fixed to 0 by default)
            - w_0: resonance frequency offset

        The model additionally gains any parameters associated with the specified Fock
        state distribution function.

        Derived parameters:
            - t_pi: Pi-time, calculated as t_pi = pi / omega
            - t_pi_2: Pi/2-time, calculated as t_pi_2 = t_pi / 2
            - f_0: Offset of resonance from zero of frequency variable in linear units

        All frequencies are in angular units.

        See also models.rabi.RabiFlop.
        """

        def __init__(
            self,
            start_excited: bool,
            sideband_index: int,
            n_max: int = 30,
        ):
            """
            :param start_excited: if True the qubit starts in the excited state
            :param sideband_index: change in motional state due to a pi-pulse starting
              from the spin ground-state.
            :param n_max: maximum Fock state used in the simulation
            """
            super().__init__(start_excited)

            self.distribution_fun = distribution_fun
            self.sideband_index = sideband_index
            self.n_max = n_max

            n = np.arange(self.n_max + 1)
            if self.start_excited:
                self._n_min = n_min = np.minimum(n, n - self.sideband_index)
                n_max = np.maximum(n, n - self.sideband_index)
            else:
                self._n_min = n_min = np.minimum(n, n + self.sideband_index)
                n_max = np.maximum(n, n + self.sideband_index)

            # sqrt(n_min!/n_max!)
            self.fact = np.exp(
                0.5 * (special.gammaln(n_min + 1) - special.gammaln(n_max + 1))
            )
            self.fact[self._n_min < 0] = 0

            # Add parameters associated with the Fock state distribution function
            spec = inspect.getfullargspec(distribution_fun)
            self.distribution_params = {
                name: copy.deepcopy(spec.annotations[name]) for name in spec.args[1:]
            }

            assert all(
                [
                    isinstance(param, ModelParameter)
                    for param in self.distribution_params.values()
                ]
            ), "Distribution function parameters must be instances of `ModelParameter`"
            assert not any(
                [
                    param_name in self.parameters.keys()
                    for param_name in self.distribution_params.keys()
                ]
            ), "Distribution parameter names must not match model parameter names"
            self.parameters.update(self.distribution_params)

        # pytype: disable=invalid-annotation
        def _func(
            self,
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
            # Rabi frequency as function of Fock state which the ion occupies
            # initially. If the ion starts in |g>, corresponds to coupling between
            # |g>|n> and |e>|n+sideband_index>. If the ion starts in |e>, corresponds
            # to coupling between |g>|n-sideband_index> and |e>|n>.
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
            t, detuning, omega = np.meshgrid(
                t_vec, detuning_vec, omega_vec, indexing="ij"
            )

            # Transition probability for individual Fock state, given by
            #     P_transition = 1/2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]
            W = np.sqrt(np.power(omega, 2) + np.power(detuning, 2))
            P_trans = (
                0.5
                * np.power(
                    np.divide(omega, W, out=np.zeros_like(W), where=(W != 0.0)), 2
                )
                * (1 - np.exp(-t / tau) * np.cos(W * t))
            )

            # Occupation probabilities of Fock states
            P_fock = self.distribution_fun(self.n_max, **kwargs)
            # Transition probability averaged over Fock state distribution
            P_trans_mean = np.sum(P_fock * P_trans, axis=-1).squeeze()

            P_e = 1 - P_trans_mean if self.start_excited else P_trans_mean

            return P_readout_g + (P_readout_e - P_readout_g) * P_e

        # pytype: enable=invalid-annotation

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
                model_parameters["n_bar"].heuristic = 1
            if "alpha" in model_parameters.keys():
                model_parameters["alpha"].heuristic = 1
            if "zeta" in model_parameters.keys():
                model_parameters["zeta"].heuristic = 1

            super().estimate_parameters(x=x, y=y, model_parameters=model_parameters)

    return LaserFlop


def make_laser_flop_freq(distribution_fun):
    """
    Fit model for Rabi flopping pulse detuning scans. See models.rabi.RabiFlopFreq.
    """
    return make_laser_flop(rabi.RabiFlopFreq, distribution_fun)


def make_laser_flop_time(distribution_fun):
    """
    Fit model for laser flopping pulse duration scans. See models.rabi.RabiFlopTime.
    """
    return make_laser_flop(rabi.RabiFlopTime, distribution_fun)


LaserFlopFreqCoherent = make_laser_flop_freq(coherent_state_probs)
LaserFlopFreqThermal = make_laser_flop_freq(thermal_state_probs)
LaserFlopFreqSqueezed = make_laser_flop_freq(squeezed_state_probs)
LaserFlopTimeCoherent = make_laser_flop_time(coherent_state_probs)
LaserFlopTimeThermal = make_laser_flop_time(thermal_state_probs)
LaserFlopTimeSqueezed = make_laser_flop_time(squeezed_state_probs)
