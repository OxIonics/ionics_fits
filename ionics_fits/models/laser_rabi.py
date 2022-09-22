import copy
import inspect
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

import ionics_fits as fits
from ionics_fits import Model, ModelParameter
from ionics_fits.utils import Array, ArrayLike


if TYPE_CHECKING:
    num_samples = float


def matrix_element_displacement_operator(m, n, alpha):
    """
    Calculate matrix element <m|D(alpha)|n> between Fock states.

    This function returns a complex number, so both magnitude
    and phase of matrix element are contained.
    TODO: Support for any of m, n and alpha being
    array_like, in which case a multidimensional array of shape
    (m.size, n.size, alpha.size) should be returned.
    """
    if not type(m) is int and type(n) is int:
        raise ValueError("Occupation numbers m and n must be integers.")
    if m < 0 or n < 0:
        raise ValueError("Occupation numbers m and n must be non-negative.")

    if m >= n:
        n_g = m
        n_l = n
        k = alpha
    else:
        n_g = n
        n_l = m
        k = -np.conjugate(alpha)

    if m == n:
        d = 1.0
    else:
        d = n_g
    i = 1
    while (n_g - i) > n_l:
        d *= n_g - i
        i += 1
    r = 1 / d

    return (
        np.sqrt(r)
        * k ** np.abs(m - n)
        * np.exp(-np.abs(alpha) ** 2 / 2)
        * eval_genlaguerre(n_l, np.abs(m - n), np.abs(alpha) ** 2)
    )


def fock_occupation_thermal_distribution(
    n: ArrayLike[int], nbar: float
) -> Array[float]:
    """
    Return occupation probability of |n> for thermal distribution.

    :param n: Fock state, array_like
    :param nbar: Average occupation number of thermal distribution
    """
    n_arr = np.atleast_1d(n)
    return 1 / (nbar + 1) * (nbar / (nbar + 1)) ** n_arr


def fock_occupation_coherent_state(n: ArrayLike[int], alpha: float) -> Array[float]:
    """
    Return occupation probability of |n> for coherent state.

    :param n: Fock state, array_like
    :param alpha: Displacement parameter of coherent state
    """
    n_arr = np.atleast_1d(n)
    if alpha == 0.0:
        P_arr = np.zeros(n_arr.size)
        P_arr[np.nonzero(n_arr == 0)] = 1.0
        return P_arr

    return np.exp(-np.abs(alpha) ** 2) * np.exp(
        n_arr * np.log(np.abs(alpha) ** 2) - gammaln(n_arr + 1)
    )


def fock_occupation_displaced_thermal_distribution(
    n: ArrayLike[int], nbar: float, alpha: float
) -> Array[float]:
    """
    Return occupation probability of |n> for "displaced" thermal distribution.

    :param n: Fock state, array_like
    :param nbar: Average occupation number of thermal distribution
    :param alpha: Parameter of displacement operator
    """
    n_arr = np.atleast_1d(n)

    # Minimum cumulative population to be contained within states included in
    # thermal distribution
    p_sum_thermal = 0.9999
    # Maximum Fock state to be included in thermal distribution, chosen such
    # that p(m <= m_max) = p_sum_thermal
    m_max = np.ceil(np.log(1 - p_sum_thermal) / np.log(nbar / (nbar + 1))) - 1
    m_arr = np.arange(m_max + 1)

    # Array to contain probabilities |<n|D(alpha)|m>|^2
    p_arr = np.zeros((n_arr.size, m_arr.size))
    for i, n in enumerate(n_arr):
        for j, m in enumerate(m_arr):
            p_arr[i, j] = np.abs(matrix_element_displacement_operator(n, m, alpha)) ** 2

    return p_arr @ fock_occupation_thermal_distribution(m_arr, nbar)


# def fock_transition_probability(t_pulse, detuning_pulse, n, W_0, eta, sb):
#     """
#     Calculate probability of transition between states |g>|n> and |e>|n+sb>.

#     :param t_pulse: Duration of probe pulse, array_like
#     :param detuning_pulse: Angular detuning of probe pulse from resonance
#         frequency of respective sideband in rad/s, array_like
#     :param n: initial motional state, array_like
#     :param W0: Base Rabi frequency corresponding to internal carrier transition
#     :param eta: Lamb-Dicke parameter for field of probe pulse
#     :param sb: Change in motional state for the sideband which the probe pulse
#         drives, -1 for rsb...
#     """
#     t_arr = np.atleast_1d(t_pulse)
#     delta_arr = np.atleast_1d(detuning_pulse)
#     # delta_arr = 2*np.pi * np.atleast_1d(detuning_pulse)
#     n_arr = np.atleast_1d(n)

#     # Array to contain effective Rabi frequency for each Fock state
#     W_arr = np.zeros(n_arr.size)
#     for i, n in enumerate(n_arr):
#         if (n + sb) >= 0:
#             W_arr[i] = W_0 * np.abs(
#                 matrix_element_displacement_operator(n + sb, n, 1j * eta))

#     # Expand arrays for broadcasting
#     t_arr_e = t_arr[:, np.newaxis, np.newaxis]
#     delta_arr_e = delta_arr[np.newaxis, :, np.newaxis]
#     W_arr_e = W_arr[np.newaxis, np.newaxis, :]

#     P = (1 / 2
#          * np.divide(W_arr_e**2,
#                      W_arr_e**2 + delta_arr_e**2,
#                      out=np.zeros((t_arr.size, delta_arr.size, W_arr.size)),
#                      where=(W_arr_e != 0.0))
#          * (1 - np.cos(np.sqrt(W_arr_e**2 + delta_arr_e**2) * t_arr_e)))
#     return P


def fock_transition_probability(
    t: ArrayLike[float],
    delta: ArrayLike[float],
    n: ArrayLike[int],
    W_0: float,
    eta: float,
    sb: int,
):
    """
    Calculate probability of transition between states |g>|n> and |e>|n+sb>.

    It is assumed that |e> lies higher in energy than |g>. This leads to
    the convention that sb = -1 corresponds to a transition that
    simultaneously raises the internal energy and extracts a motional quantum.

    The array_likes t, delta and n are passed directly to numpy
    functions, which means that standard broadcasting rules apply for these
    variables.

    :param t: Duration (in s) of interaction between atom and driving field
    :param delta: Angular detuning (in rad/s) of probe pulse from resonance
        frequency of respective sideband
    :param n: Fock state that ion initially occupies
    :param W_0: Base Rabi frequency corresponding to internal transition
    :param eta: Lamb-Dicke parameter for driving field
    :param sb: Change in motional state for the sideband which the probe pulse
        drives, -1 for rsb...
    """
    if type(sb) is not int:
        raise ValueError("Sideband variable must be integer.")

    t_arr = np.atleast_1d(t)
    delta_arr = np.atleast_1d(delta)
    n_arr = np.atleast_1d(n)

    # Array to contain effective Rabi frequency for each Fock state
    W_arr = np.zeros(n_arr.shape)
    for i in range(n_arr.shape[-1]):
        # If the array_like parameter n has been passed correctly, all
        # elements of the slice should be equal, so just pick one of them
        n = n_arr[..., i].flat[0]
        if (n + sb) >= 0:
            W_arr[..., i] = W_0 * np.abs(
                matrix_element_displacement_operator(n + sb, n, 1j * eta)
            )

    P = (
        1
        / 2
        * np.divide(
            W_arr**2,
            W_arr**2 + delta_arr**2,
            out=np.zeros(
                delta_arr.shape if delta_arr.size >= n_arr.size else n_arr.shape
            ),
            where=(W_arr != 0.0),
        )
        * (1 - np.cos(np.sqrt(W_arr**2 + delta_arr**2) * t_arr))
    )
    return P


class LaserFlop(Model):
    def __init__(self, sb, P_e_initial):
        """
        :param sb: Sideband that is being driven, see function
            fock_transition_probability
        :param P_e_initial: initial population of higher-energy state, must
            be either 0 or 1
        """
        self.sb = sb
        if P_e_initial not in [0.0, 1.0]:
            raise ValueError(
                "Initial population of excited state must be either 0 or 1."
            )
        self.P_e_initial = P_e_initial
        # Minimum cumulative population to be contained within states that
        # will be included when averaging over motional distribution
        self.p_sum_fock = 0.999

        self.parameters = dict()
        spec = inspect.getfullargspec(self._fock_observation_probability)
        param_names = spec.args[3:]
        for name in param_names:
            if not isinstance(spec.annotations[name], ModelParameter):
                raise ValueError(
                    "Model parameters must be instances of 'ModelParameter'."
                )
        self.parameters.update(
            {name: copy.deepcopy(spec.annotations[name]) for name in param_names}
        )
        setattr(self, "func_obs_param_names", param_names)

        spec = inspect.getfullargspec(self._fock_occupation_probability)
        param_names = spec.args[2:]
        for name in param_names:
            if not isinstance(spec.annotations[name], ModelParameter):
                raise ValueError(
                    "Model parameters must be instances of 'ModelParameter'."
                )
        self.parameters.update(
            {name: copy.deepcopy(spec.annotations[name]) for name in param_names}
        )
        setattr(self, "func_occ_param_names", param_names)

    def func(
        self,
        x: Tuple[
            Array[("num_samples",), np.float64],
            Array[("num_samples",), np.float64],
        ],
        param_values: Dict[str, float],
    ):
        """
        Calculate observation probability averaged over motional distribution.

        :param x: Tuple (duration, delta) of two array_likes containing
            nominal duration and angular detuning of probe pulse from
            resonance frequency of sideband used by this model
        """
        func_obs_dict = {name: param_values[name] for name in self.func_obs_param_names}
        func_occ_dict = {name: param_values[name] for name in self.func_occ_param_names}

        n_max = self.calculate_maximum_fock(**func_occ_dict)
        n_arr = np.arange(n_max + 1)
        P_obs_fock = self._fock_observation_probability(x, n_arr, **func_obs_dict)
        P_occ_fock = self._fock_occupation_probability(n_arr, **func_occ_dict)[
            np.newaxis, np.newaxis, :
        ]

        P = np.sum(P_occ_fock * P_obs_fock, axis=2)

        return np.squeeze(P)

    def calculate_maximum_fock(self):
        raise NotImplementedError

    def _fock_observation_probability(
        self,
        x: Tuple[
            Array[("num_samples",), np.float64],
            Array[("num_samples",), np.float64],
        ],
        n: Array[int],
        W_0: ModelParameter(lower_bound=0.0),
        eta: ModelParameter(lower_bound=0.0),
        P_readout_e: ModelParameter(lower_bound=0.0, upper_bound=1.0),
        P_readout_g: ModelParameter(lower_bound=0.0, upper_bound=1.0),
        t_dead: ModelParameter(lower_bound=0.0),
    ):
        t_grid, delta_grid, n_grid = np.meshgrid(x[0], x[1], n, indexing="ij")
        if self.P_e_initial == 0.0:
            P_e = fock_transition_probability(
                t_grid, delta_grid, n_grid, W_0, eta, self.sb
            )
        else:
            P_e = 1 - fock_transition_probability(
                t_grid, delta_grid, n_grid, W_0, eta, -self.sb
            )

        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    def _fock_occupation_probability(
        self, n: Array[int], *args: ModelParameter()
    ) -> Array[float]:
        raise NotImplementedError


class LaserFlopThermal(LaserFlop):
    def _fock_occupation_probability(
        self,
        n: Array[int],
        nbar: ModelParameter(lower_bound=0.0),
    ) -> Array[float]:
        return fock_occupation_thermal_distribution(n, nbar)

    def calculate_maximum_fock(self, nbar: float) -> int:
        """
        Calculate maximum occupation number to be included in distribution.

        This method finds the occupation number n_max such that the total
        population contained in all states with n <= n_max is at least
        self.p_sum_fock.
        """
        if nbar == 0.0:
            return 0

        return int(np.ceil(np.log(1 - self.p_sum_fock) / np.log(nbar / (nbar + 1))) - 1)


class LaserFlopCoherent(LaserFlop):
    def _fock_occupation_probability(
        self,
        n: Array[int],
        alpha: ModelParameter(lower_bound=0.0),
    ) -> Array[float]:
        return fock_occupation_coherent_state(n, alpha)

    def calculate_maximum_fock(self, alpha: float) -> int:
        """
        Calculate maximum occupation number to be included in distribution.

        This method finds the occupation number n_max such that the total
        population contained in all states with n <= n_max is at least
        self.p_sum_fock.

        # TODO: find an accurate way of calculating this
        """
        if alpha == 0.0:
            return 0

        n_i = 0
        while True:
            P_arr = fock_occupation_coherent_state(np.arange(n_i + 1), alpha)
            if np.sum(P_arr) >= self.p_sum_fock:
                break
            n_i += 5
        return n_i

    # def estimate_parameters(
    #     self,
    #     x: Array[("num_samples",), np.float64],
    #     y: Array[("num_samples",), np.float64],
    #     model_parameters: Dict[str, ModelParameter],
    # ):
    #     self._estimate_internal_parameters()
    #     self._esimate_motional_parameters()

    # def _estimate_internal_parameters(
    #     self,
    #     x: Array[("num_samples",), np.float64],
    #     y: Array[("num_samples",), np.float64],
    #     model_parameters: Dict[str, ModelParameter],
    # ):
    #     raise NotImplementedError


class LaserFlopTime(LaserFlop):
    def __init__(self, sb, P_e_initial):
        super().__init__(sb, P_e_initial)

        self.parameters["delta"] = ModelParameter()

        self.parameters["delta"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["W_0"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ):
        """
        Calculate observation probability averaged over motional distribution.

        :param x: Nominal duration of probe pulse, not accounting for dead time
        """
        t = x - param_values["t_dead"]
        delta = param_values["delta"]
        return super().func((t, delta), param_values)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """
        Sets initial values for model parameters based on heuristics. Typically
        called during `Fitter.fit`.

        Heuristic results should be stored in :param model_parameters: using the
        `ModelParameter`'s `initialise` method. This ensures that all information passed
        in by the user (fixed values, initial values, bounds) is used correctly.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data

        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        model_parameters["delta"].initialise(0.0)
        model_parameters["eta"].initialise(0.0)
        if self.P_e_initial == 0.0:
            model_parameters["P_readout_g"].initialise(y[0])
            # FIXME: this estimate for P_readout_e is not quite valid, but reasonable
            model_parameters["P_readout_e"].initialise(1 - y[0])
        else:
            # FIXME: this estimate for P_readout_g is not quite valid, but reasonable
            model_parameters["P_readout_g"].initialise(1 - y[0])
            model_parameters["P_readout_e"].initialise(y[0])
        model_parameters["t_dead"].initialise(0.0)

        sinusoid = fits.models.Sinusoid()
        sinusoid.parameters["phi"].fixed_to = np.pi / 2 if y[0] > 0.5 else 3 * np.pi / 2
        fit = fits.NormalFitter(x, y, sinusoid)
        model_parameters["W_0"].initialise(fit.values["omega"])


# class LaserFlopFreq(LaserFlop):
#     def __init__(self):
#         super().__init__()

#         self.parameters["t_pulse"] = ModelParameter(lower_bound=0.0)

#     def func(
#         self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
#     ):
#         t = param_values["t_pulse"] - param_values["t_dead"]
#         delta_pulse = param_values["delta_pulse"]
#         return super().func((t, delta_pulse), param_values)


class LaserFlopTimeThermal(LaserFlopTime, LaserFlopThermal):
    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        super().estimate_parameters(x, y, model_parameters)
        model_parameters["nbar"].initialise(0.0)


class LaserFlopTimeCoherent(LaserFlopTime, LaserFlopCoherent):
    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        super().estimate_parameters(x, y, model_parameters)
        model_parameters["alpha"].initialise(0.0)
