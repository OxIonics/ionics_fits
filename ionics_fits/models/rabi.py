from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .sinc import Sinc2
from .sinusoid import Sinusoid
from .. import Model, ModelParameter, NormalFitter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class RabiFlop(Model):
    """
    Base class for damped Rabi flops

    The model function of this class calculates the probability P(E) of
    observing some readout event "E" after a system comprised of two states,
    labelled |g> and |e>, has interacted with a sinusoidal driving field. This
    probability is given by
        P(E) = P_readout_g + [P_readout_e - P_readout_g] * P_e
    where P_e is the time-dependent population in state |e>, while P_readout_g
    and P_readout_e denote the readout levels of the states. Explicitly,
    P_readout_i is the conditional probability P(E|i) that the readout event
    is observed given that the particle occupies state |i>.

    This model requires that the particle starts out either in |g> or in |e>,
    specified by passing :param start_excited: to the __init__ method. The
    probability of transition from one state to the other may then be
    calculated as
        P_trans = 1 / 2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]
    where
        - t is the duration of interaction between qubit and driving field
        - W = sqrt(omega^2 + delta^2)
        - delta is the detuning of driving field from resonance frequency of qubit
        - omega is the Rabi frequency
        - tau is the decay time constant

    The event "E" may for example be defined as "ion bright", but this is not
    required by the model. The user only needs to provide the state which the
    particle initially occupies.

    This class does not support fitting directly, use one of the subclasses
    :class RabiFlopFreq: or :class RabiFlopTime: instead.

    Independent variables:
        - t_pulse: Nominal pulse duration of driving field
        - freq: Frequency of driving field

    Model parameters:
        - P_readout_e: Probability P(E|e) of observation of readout event
            when particle occupies state |e> (fixed to 1 by default)
        - P_readout_g: Probability P(E|g) of observation of readout event
            when particle occupies state |g> (fixed to 0 by default)
        - omega: Rabi frequency
        - tau: Decay time constant (fixed to infinity by default)
        - t_dead: Dead time for driving field (fixed to 0 by default). The
            true duration of interaction is calculated as t = max(0, t_pulse - t_dead).
        - freq_offset: Offset of resonance from zero of frequency variable.
            The true detuning is calculated as delta = freq - freq_offset.

    Derived parameters:
        - t_pi: Pi-time in the absence of dead-time,
            calculated as t_pi = pi / omega
        - t_pi_2: Pi/2-time in the absence of dead-time,
            calculated as t_pi_2 = t_pi / 2

    All frequencies and detunings are in angular units.
    """

    def __init__(self, start_excited):
        super().__init__()
        self.start_excited = start_excited

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Tuple[
            Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
        ],
        P_readout_e: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_to=1.0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        P_readout_g: ModelParameter(
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_to=0.0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        omega: ModelParameter(lower_bound=0.0),
        tau: ModelParameter(
            lower_bound=0.0,
            fixed_to=np.inf,
        ),
        t_dead: ModelParameter(
            lower_bound=0.0,
            fixed_to=0.0,
        ),
        freq_offset: ModelParameter(fixed_to=0.0),
    ) -> Array[("num_samples",), np.float64]:
        """
        Return the probability of observation of readout event.

        :param x: Tuple (t_pulse, freq) of ndarrays containing nominal
            pulse duration and frequency of driving field. They must have
            shapes such that they are broadcastable.
        """
        t = np.clip(x[0] - t_dead, a_min=0.0, a_max=None)
        delta = x[1] - freq_offset
        W = np.sqrt(omega**2 + delta**2)

        P_trans = (
            1
            / 2
            * np.divide(omega**2, W**2, out=np.zeros_like(W), where=(W != 0.0))
            * (1 - np.exp(-t / tau) * np.cos(W * t))
        )
        P_e = 1 - P_trans if self.start_excited else P_trans

        return P_readout_g + (P_readout_e - P_readout_g) * P_e

    # pytype: enable=invalid-annotation

    @staticmethod
    def calculate_derived_params(
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

        return derived_params, derived_uncertainties


class RabiFlopFreq(RabiFlop):
    def __init__(self, start_excited=True):
        super().__init__(start_excited)

        self.parameters["t_pulse"] = ModelParameter(lower_bound=0.0)

        self.parameters["t_pulse"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["freq_offset"].scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        t_pulse = param_values["t_pulse"]
        del param_values["t_pulse"]
        return super()._func((t_pulse, x), **param_values)
        # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """
        Sets initial values for model parameters based on heuristics.

        Typically called during `Fitter.fit`. Heuristic results should stored
        in :param model_parameters: using the
        `ModelParameter`'s `initialise` method. This ensures that all information passed
        in by the user (fixed values, initial values, bounds) is used correctly.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        # TODO: Automatically estimate readout levels
        model_parameters["P_readout_e"].initialise(1.0)
        model_parameters["P_readout_g"].initialise(0.0)
        model_parameters["t_dead"].initialise(0.0)
        model_parameters["tau"].initialise(np.inf)

        # There isn't a simple analytic form for the Fourier transform of a Rabi
        # flop in the general case. However in the low pulse area limit (and
        # ignoring decay etc) the Rabi flop function tends to the sinc^2 function:
        #   (omega * t / 2)^2 * sinc^2(delta * t / 2)
        # NB np.sinc(x) = np.sin(pi * x) / (pi * x)
        # This heuristic breaks down when: omega * t_pulse ~ pi
        model = Sinc2()
        model.parameters["y0"].fixed_to = 1.0
        fit = NormalFitter(x, y, model)

        a = np.abs(fit.values["a"])
        t_pulse = model_parameters["t_pulse"].initialise(2 * fit.values["w"])
        model_parameters["omega"].initialise(2 * np.sqrt(a) / t_pulse)
        model_parameters["freq_offset"].initialise(-fit.values["x0"])


class RabiFlopTime(RabiFlop):
    def __init__(self, start_excited=True):
        super().__init__(start_excited)

        self.parameters["freq"] = ModelParameter()

        self.parameters["freq"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["freq_offset"].scale_func = (
            lambda x_scale, y_scale, _: 1 / x_scale
        )

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        freq = param_values["freq"]
        del param_values["freq"]
        return super()._func((x, freq), **param_values)
        # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """
        Set initial values for model parameters based on heuristics.

        Typically called during `Fitter.fit`.

        Heuristic results should stored in :param model_parameters: using the
        `ModelParameter`'s `initialise` method. This ensures that all information passed
        in by the user (fixed values, initial values, bounds) is used correctly.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        # TODO: Automatically estimate readout levels
        model_parameters["P_readout_e"].initialise(1.0)
        model_parameters["P_readout_g"].initialise(0.0)
        model_parameters["t_dead"].initialise(0.0)
        model_parameters["tau"].initialise(np.inf)

        model = Sinusoid()
        model.parameters["phi"].fixed_to = (
            np.pi / 2 if self.start_excited else 3 * np.pi / 2
        )
        fit = NormalFitter(x, y, model)

        W = fit.values["omega"]
        omega = np.sqrt(2 * fit.values["a"]) * W
        # Avoid divide by zero errors from numerical noise when delta ~= 0
        delta = 0.0 if omega >= W else np.sqrt(W**2 - omega**2)

        model_parameters["omega"].initialise(omega)
        model_parameters["freq_offset"].initialise(0.0)
        model_parameters["freq"].initialise(delta)
