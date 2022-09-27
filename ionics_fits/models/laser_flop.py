from typing import Callable, Dict, TYPE_CHECKING, Tuple

import numpy as np
from scipy.special import eval_genlaguerre, factorial, gammaln

from ionics_fits.common import Array, ModelParameter
import ionics_fits as fits


if TYPE_CHECKING:
    num_samples = float
    num_fock_states = float


def thermal_state_probs(
    n_max: int, n_bar: ModelParameter(lower_bound=0)
) -> Array[("num_fock_states",), np.float64]:
    """ Returns an array with the Fock state occupation probabilities for a thermal
    state of mean occupancy :param n_bar:, truncated at a maximum Fock state of |n_max>
    """
    n = np.arange(n_max + 1, dtype=int)
    return 1 / (n_bar + 1) * np.power((n_bar / (n_bar + 1)), n)


def coherent_state_probs(n_max: int, alpha: ModelParameter(lower_bound=0)
) -> Array[("num_fock_states",), np.float64]:
    """Returns an array with the Fock state occupation probabilities for a coherent
    state described by :param alpha:, truncated at a maximum Fock state of |n_max>
    """
    n = np.arange(n_max + 1, dtype=int)
    # return np.exp(-0.5*alpha**2) * np.power(alpha, n) / np.sqrt(factorial(n))
    return np.exp(-np.abs(alpha) ** 2) * np.exp(
        n * np.log(np.abs(alpha) ** 2) - gammaln(n + 1)
    )

class LaserFlop(fits.Model):
    """Add note here that we split the dynamics into decoupled two-state systems etc"""

    def __init__(
        self,
        prob_fun: Callable,
        sideband: int,
        n_max: int,
        prepare_excited: bool = True,
    ):
        """ """
        super().__init__()
        self._extend_parameters(prob_fun, skip=1)

        if sideband not in [-1, 0, +1]:
            raise ValueError(f"Unsupported sideband order {sideband}")

        self.sideband = sideband
        self.n_max = n_max
        self.prepare_excited = True
        self.prob_fun = prob_fun

    def _func(
        self,
        x: Tuple[
            Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
        ],
        omega_0: ModelParameter(lower_bound=0.0),
        eta: ModelParameter(lower_bound=0.0),
        P_readout_e: ModelParameter(lower_bound=0.0, upper_bound=1.0, fixed_to=1),
        P_readout_g: ModelParameter(lower_bound=0.0, upper_bound=1.0, fixed_to=0),
        t_dead: ModelParameter(lower_bound=0.0, fixed_to=0),
        **kwargs,
    ):
        n = np.arange(self.n_max + 1)
        n_min = n - np.abs(self.sideband)
        n_max = n + np.abs(self.sideband)

        omega_vec = (
            omega_0
            * np.exp(-0.5 * eta**2)
            * eval_genlaguerre(n_min, np.abs(self.sideband), eta**2)
        )

        if self.sideband != 0:
            omega_vec = np.divide(
                omega_vec * np.power(eta, np.abs(self.sideband)),
                np.sqrt(n_max),
                out=np.zeros_like(omega_vec),
                where=n_max > 0,
            )

        t_vec = np.clip(np.asarray(x[0]) - t_dead, a_min=0, a_max=None)
        detuning_vec = np.asarray(x[1])
        t, detuning, omega = np.meshgrid(t_vec, detuning_vec, omega_vec, indexing="ij")

        omega_eff = np.sqrt(np.power(omega, 2) + np.power(detuning, 2))

        P_flip = np.power(
            np.divide(
                omega * np.sin(omega_eff * t / 2),
                omega_eff,
                out=np.zeros_like(omega),
                where=omega_eff > 0,
            ),
            2,
        )
        initial_state = self.prob_fun(self.n_max, **kwargs)
        P_i = np.tile(initial_state, (omega.shape[0], omega.shape[1], 1))
        P_e = np.sum(P_i * P_flip, axis=-1).squeeze()

        P = P_readout_g + (P_readout_e - P_readout_g) * P_e

        if not self.prepare_excited:
            P = 1 - P_e

        return P

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """ Estimates the common parameters. Subcalsses should extend this method to add
        heuristics for parameters associated with time / detuning scans and particular
        Fock state distributions.
        """
        model_parameters["eta"].initialise(0.0)
        model_parameters["P_readout_g"].initialise(0.0)
        model_parameters["P_readout_e"].initialise(1.0)
        model_parameters["t_dead"].initialise(0.0)


class LaserFlopTime(LaserFlop):
    def __init__(
        self,
        prob_fun: Callable,
        sideband: int,
        n_max: int,
        prepare_excited: bool = True,
    ):
        super().__init__(prob_fun, sideband, n_max, prepare_excited)

        self.parameters["delta"] = ModelParameter()

        self.parameters[
            "delta"
        ].scale_func = lambda x_scale, y_scale, _: None  # 1 / x_scale
        self.parameters["omega_0"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ):
        """
        Calculate observation probability averaged over motional distribution.
        :param x: Nominal duration of probe pulse, not accounting for dead time
        """
        param_values = param_values.copy()
        t = x - param_values["t_dead"]
        delta = param_values.pop("delta")  # TODO: think more about handling of this...
        return super().func((t, delta), param_values)


class LaserFlopTimeThermal(LaserFlopTime):
    def __init__(self, sideband, n_max, prepare_excited: bool = True):
        super().__init__(
            thermal_state_probs,
            sideband,
            n_max,
            prepare_excited,
        )

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        super().estimate_parameters(x, y, model_parameters)
        model_parameters["delta"].initialise(0.0)

        sinusoid = fits.models.Sinusoid()
        sinusoid.parameters["tau"].fixed_to = None
        fit = fits.NormalFitter(x, y, sinusoid)
        model_parameters["omega_0"].initialise(fit.values["omega"])

        tau = fit.values["tau"] * fit.values["omega"]
        n_bar = 2*np.log(700 / (tau - 15))
        model_parameters["n_bar"].initialise(n_bar)


class LaserFlopTimeCoherent(LaserFlopTime):
    def __init__(self, sideband, n_max, prepare_excited: bool = True):
        super().__init__(
            coherent_state_probs,
            sideband,
            n_max,
            prepare_excited,
        )

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        super().estimate_parameters(x, y, model_parameters)
        model_parameters["delta"].initialise(0.0)

        if model_parameters["omega_0"].get_initial_value() is None:
            sinusoud = fits.models.Sinusoid()
            model.estimate_parameters(x, y, model.parameters)
            omega_0 = model.parameters["omega"].get_initial_value()
            model_parameters["omega_0"].initialise(omega_0)

        # TODO: this is a bit fragile and could be improved.
        model_parameters["alpha"].initialise(0.)
