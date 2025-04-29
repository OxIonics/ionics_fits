import copy
import inspect
from typing import TYPE_CHECKING, Callable, Dict, Tuple

import numpy as np
from scipy import special

from ..common import TX, TY, ModelParameter
from ..utils import Array, scale_invariant, scale_undefined, scale_y
from .quantum_phys import (
    coherent_state_probs,
    displaced_thermal_state_probs,
    squeezed_state_probs,
    thermal_state_probs,
)
from .rabi import RabiFlop, RabiFlopFreq, RabiFlopTime
from .utils import param_like

if TYPE_CHECKING:
    num_fock_states = float


class LaserFlop(RabiFlop):
    r"""Base class for damped Rabi flopping with finite Lamb-Dicke parameter.

    This model calculates measurement outcomes for systems containing two internal
    states and moving in a 1D harmonic potential that undergo damped Rabi oscillations,
    defined by::

        P = P_readout_g + (P_readout_e - P_readout_g) * P_e

    where ``P_e`` is the (time-dependent) population in the excited state and
    ``P_readout_g`` and ``P_readout_e`` are the readout levels (measurement outcomes
    when the qubit is in one state).

    This class does not support fitting directly; use one of the subclasses instead.
    Subclasses must inherit from this class and a suitable
    :class:`~ionics_fits.models.rabi.RabiFlop` subclass, such as
    :class:`~ionics_fits.models.rabi.RabiFlopFreq` or
    :class:`~ionics_fits.models.rabi.RabiFlopTime`.

    The model requires that the spin state of the system starts out entirely in one
    of the ground or excited states, specified using :meth:`__init__`\'s
    ``start_excited`` parameter. It further assumes that the motional part of the system
    starts out in a distribution over different Fock states, described by the specified
    ``distribution_fun``.

    Independent variables:

        * ``t_pulse``: duration of driving pulse including dead time. The duration of
          the interaction is given by ``t = max(0, t_pulse - t_dead)``.
        * ``w``: frequency of driving pulse relative to the reference frequency
          ``w_0``\, given by ``delta = w - w_0``

    The model additionally gains any parameters associated with the specified Fock
    state distribution function.

    Derived parameters are inherited from :class:`~ionics_fits.models.rabi.RabiFlop`\'s
    :func:`~ionics_fits.models.rabi.RabiFlop.calculate_derived_params` method.

    All frequencies are in angular units.

    See also :class:`~ionics_fits.models.rabi.RabiFlop`.
    """

    def __init__(
        self,
        distribution_fun: Callable[..., Array[("num_fock_states",), np.float64]],
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        r"""
        :param distribution_fun: function returning an array of Fock state occupation
          probabilities. The distribution function's first argument should be the
          maximum Fock state to include in the simulation (the returned array has
          ``n_max + 1`` elements). Subsequent arguments should be
          :class:`~ionics_fits.common.ModelParameter`\s used to parametrise the
          distribution.
        :param start_excited: if ``True`` the qubit starts in the excited state
        :param sideband_index: change in motional state due to a pi-pulse starting
          from the spin ground-state.
        :param n_max: maximum Fock state used in the simulation
        """
        super().__init__(start_excited=start_excited)

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
        eta: ModelParameter(lower_bound=0.0, scale_func=scale_invariant),
        omega: ModelParameter(lower_bound=0.0, scale_func=scale_undefined),
        tau: ModelParameter(
            lower_bound=0.0, fixed_to=np.inf, scale_func=scale_undefined
        ),
        t_dead: ModelParameter(
            lower_bound=0.0, fixed_to=0.0, scale_func=scale_undefined
        ),
        w_0: ModelParameter(scale_func=scale_undefined),
        # Fock state distribution function parameters
        **kwargs: ModelParameter,
    ) -> TY:
        """Return measurement probability.

        :param x: tuple of ``(t_pulse, w)``. Subclasses should override func to
            map this onto the appropriate input data.
        :param P_readout_e: excited state readout level
        :param P_readout_g: ground state readout level
        :param eta: Lamb-Dicke parameter
        :param omega: carrier Rabi frequency
        :param tau: decay time constant
        :param t_dead: dead time
        :param w_0: resonance frequency offset
        """
        # Rabi frequency as function of Fock state which the ion occupies
        # initially. If the ion starts in |g>, corresponds to coupling between
        # |g>|n> and |e>|n+sideband_index>. If the ion starts in |e>, corresponds
        # to coupling between |g>|n-sideband_index> and |e>|n>.
        omega_vec = (
            omega
            * np.exp(-0.5 * eta**2)
            * eta ** np.abs(self.sideband_index)
            * self.fact
            * special.eval_genlaguerre(self._n_min, np.abs(self.sideband_index), eta**2)
        )

        t_vec = np.clip(x[0] - t_dead, a_min=0, a_max=None)
        detuning_vec = x[1]
        t, detuning, omega = np.meshgrid(t_vec, detuning_vec, omega_vec, indexing="ij")

        # Transition probability for individual Fock state, given by
        #     P_transition = 1/2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]
        W = np.sqrt(omega**2 + detuning**2)
        P_trans = (
            0.5
            * (np.divide(omega, W, out=np.zeros_like(W), where=(W != 0.0)) ** 2)
            * (1 - np.exp(-t / tau) * np.cos(W * t))
        )

        # Occupation probabilities of Fock states
        P_fock = self.distribution_fun(self.n_max, **kwargs)
        # Transition probability averaged over Fock state distribution
        P_trans_mean = np.sum(P_fock * P_trans, axis=-1).squeeze()

        P_e = 1 - P_trans_mean if self.start_excited else P_trans_mean

        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    # pytype: enable=invalid-annotation

    def estimate_parameters(self, x: TX, y: TY):
        # Pick sensible starting values which are usually good enough for the fit to
        # converge from.
        if (
            self.parameters["omega"].has_user_initial_value()
            and not self.parameters["eta"].has_user_initial_value()
        ):
            # use the rabi heuristic to find eta * omega
            eta_heuristic = True
            omega_param = self.parameters["omega"]

            self.parameters["omega"] = param_like(omega_param)
            self.parameters["eta"].heuristic = 0  # avoid exception due to no value

            omega_lower = self.parameters["omega"].lower_bound or 0
            omega_upper = self.parameters["omega"].upper_bound or np.inf

            eta_lower = self.parameters["eta"].lower_bound or 0
            eta_upper = self.parameters["eta"].upper_bound or np.inf

            self.parameters["omega"].fixed_to = None
            self.parameters["omega"].user_estimate = None
            self.parameters["omega"].lower_bound = omega_lower * eta_lower
            self.parameters["omega"].upper_bound = omega_upper * eta_upper
        else:
            eta_heuristic = False

        if "n_bar" in self.parameters.keys():
            self.parameters["n_bar"].heuristic = 1
        if "alpha" in self.parameters.keys():
            self.parameters["alpha"].heuristic = 1
        if "zeta" in self.parameters.keys():
            self.parameters["zeta"].heuristic = 1

        super().estimate_parameters(x=x, y=y)

        if eta_heuristic:
            eta_omega = self.parameters["omega"].get_initial_value()
            self.parameters["eta"].heuristic = (
                eta_omega / omega_param.get_initial_value()
            )
            self.parameters["omega"] = omega_param

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        # use eta * omega for calculating pi times etc
        fitted_params = dict(fitted_params)
        fit_uncertainties = dict(fit_uncertainties)

        omega = fitted_params.pop("omega")
        eta = fitted_params.pop("eta")

        omega_uncert = fit_uncertainties.pop("omega")
        eta_uncert = fit_uncertainties.pop("eta")

        fitted_params["omega"] = eta * omega
        fit_uncertainties["omega"] = np.sqrt(omega_uncert**2 + eta_uncert**2)

        return super().calculate_derived_params(
            x=x,
            y=y,
            fitted_params=fitted_params,
            fit_uncertainties=fit_uncertainties,
        )


class LaserFlopFreqCoherent(LaserFlop, RabiFlopFreq):
    """Fit model for Rabi flopping pulse detuning scans when the motional degree of
    freedom starts in a coherent state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=coherent_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopFreqThermal(LaserFlop, RabiFlopFreq):
    """Fit model for Rabi flopping pulse detuning scans when the motional degree of
    freedom starts in a thermal state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=thermal_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopFreqSqueezed(LaserFlop, RabiFlopFreq):
    """Fit model for Rabi flopping pulse detuning scans when the motional degree of
    freedom starts in a squeezed state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=squeezed_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopFreqDisplacedThermal(LaserFlop, RabiFlopFreq):
    """Fit model for Rabi flopping pulse detuning scans when the motional degree of
    freedom starts in a displaced thermal state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=displaced_thermal_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopTimeCoherent(LaserFlop, RabiFlopTime):
    """Fit model for Rabi flopping pulse duration scans when the motional degree of
    freedom starts in a coherent state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=coherent_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopTimeThermal(LaserFlop, RabiFlopTime):
    """Fit model for Rabi flopping pulse duration scans when the motional degree of
    freedom starts in a thermal state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=thermal_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopTimeSqueezed(LaserFlop, RabiFlopTime):
    """Fit model for Rabi flopping pulse duration scans when the motional degree of
    freedom starts in a squeezed state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=squeezed_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )


class LaserFlopTimeDisplacedThermal(LaserFlop, RabiFlopTime):
    """Fit model for Rabi flopping pulse duration scans when the motional degree of
    freedom starts in a displaced thermal state.
    """

    def __init__(
        self,
        start_excited: bool,
        sideband_index: int,
        n_max: int = 30,
    ):
        super().__init__(
            distribution_fun=displaced_thermal_state_probs,
            start_excited=start_excited,
            sideband_index=sideband_index,
            n_max=n_max,
        )
