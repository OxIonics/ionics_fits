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

    This model calculates the measurement probability for damped Rabi flops on
    a system with states |g> and |e>, given by
        P = P_readout_g + (P_readout_e - P_readout_g) * P_e
    where P_e is the time-dependent population in the excited state, while
    P_readout_g and P_readout_e denote the individual readout levels.

    The model requires that the system starts out in either |g> or |e>,
    specified by passing :param:`start_excited` to :meth:`__init__`. The
    probability of transition from one state to the other may then be
    calculated as
        P_trans = 1 / 2 * omega^2 / W^2 * [1 - exp(-t / tau) * cos(W * t)]
    where
        - t is the duration of interaction between qubit and driving field
        - W = sqrt(omega^2 + delta^2)
        - delta is the detuning of the driving field from the resonance frequency
        - omega is the Rabi frequency
        - tau is the decay time constant

    This class does not support fitting directly, use one of the subclasses
    :class RabiFlopFreq: or :class RabiFlopTime: instead.

    Independent variables:
        - t_pulse: Duration of driving pulse including dead time. The true duration of
            interaction is calculated as t = max(0, t_pulse - t_dead).
        - w: Variable determining frequency of driving pulse. This does not have to be
            the absolute frequency, but may instead be measured relative to some
            arbitrary reference frequency. The detuning from resonance is calculated
            as delta = w - w_0.

    Model parameters:
        - P_readout_e: Readout level for state |e> (fixed to 1 by default)
        - P_readout_g: Readout level for state |g> (fixed to 0 by default)
        - omega: Rabi frequency
        - tau: Decay time constant (fixed to infinity by default)
        - t_dead: Dead time (fixed to 0 by default)
        - w_0: Offset of resonance from zero of frequency variable

    Derived parameters:
        - t_pi: Pi-time, calculated as t_pi = pi / omega
        - t_pi_2: Pi/2-time, calculated as t_pi_2 = t_pi / 2

    All frequencies are in angular units.
    """

    def __init__(self, start_excited: bool):
        super().__init__()
        self.start_excited = start_excited

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
        tau: ModelParameter(lower_bound=0.0, fixed_to=np.inf),
        t_dead: ModelParameter(lower_bound=0.0, fixed_to=0.0),
        w_0: ModelParameter(),
    ) -> Array[("num_samples",), np.float64]:
        """
        Return measurement probability.

        :param x: Tuple (t_pulse, w) of ndarrays containing pulse duration and
            angular frequency of driving field. They must have shapes such
            that they are broadcastable.
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
        """
        Return measurement probability as function of pulse frequency.

        :param x: Angular frequency
        """
        t_pulse = param_values.pop("t_pulse")
        return super()._func((t_pulse, x), **param_values)
        # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        # By default, we assume that the readout level for |g> is low and the
        # one for |e> is high. If this is not the case, `user_estimate` has to be
        # provided instead.
        model_parameters["P_readout_g"].heuristic = 0.0
        model_parameters["P_readout_e"].heuristic = 1.0
        model_parameters["t_dead"].heuristic = 0.0
        model_parameters["tau"].heuristic = np.inf

        # There isn't a simple analytic form for the Fourier transform of a Rabi
        # flop in the general case. However in the low pulse area limit (and
        # ignoring decay etc) the Rabi flop function tends to the sinc^2 function:
        #   (omega * t / 2)^2 * sinc^2(delta * t / 2)
        # NB np.sinc(x) = np.sin(pi * x) / (pi * x)
        # This heuristic breaks down when: omega * t_pulse ~ pi
        model = Sinc2()
        model.parameters["y0"].fixed_to = (
            model_parameters["P_readout_e"].get_initial_value()
            if self.start_excited
            else model_parameters["P_readout_g"].get_initial_value()
        )
        fit = NormalFitter(x, y, model)

        model_parameters["t_pulse"].heuristic = 2 * fit.values["w"]
        t_pulse = model_parameters["t_pulse"].get_initial_value()
        model_parameters["omega"].heuristic = (
            2 * np.sqrt(np.abs(fit.values["a"])) / t_pulse
        )
        model_parameters["w_0"].heuristic = fit.values["x0"]


class RabiFlopTime(RabiFlop):
    def __init__(self, start_excited: bool):
        super().__init__(start_excited)

        self.parameters["w"] = ModelParameter()

        self.parameters["w"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["omega"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["tau"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        self.parameters["t_dead"].scale_func = lambda x_scale, y_scale, _: x_scale
        self.parameters["w_0"].scale_func = lambda x_scale, y_scale, _: 1 / x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """
        Return measurement probability as function of pulse duration.

        :param x: Pulse duration
        """
        w = param_values.pop("w")
        return super()._func((x, w), **param_values)
        # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        # By default, we assume that the readout level for |g> is low and the
        # one for |e> is high. If this is not the case, `user_estimate` has to be
        # provided instead.
        model_parameters["P_readout_g"].heuristic = 0.0
        model_parameters["P_readout_e"].heuristic = 1.0
        model_parameters["t_dead"].heuristic = 0.0
        model_parameters["tau"].heuristic = np.inf

        model = Sinusoid()
        model.parameters["phi"].fixed_to = (
            np.pi / 2 if self.start_excited else 3 * np.pi / 2
        )
        fit = NormalFitter(x, y, model)
        W = fit.values["omega"]
        omega = np.sqrt(2 * fit.values["a"]) * W
        # Prevent delta < 0 from numerical noise
        delta = 0.0 if omega >= W else np.sqrt(W**2 - omega**2)

        model_parameters["omega"].heuristic = omega
        # Transition probability only depends on difference w - w_0, so don't float
        # both parameters simultaneously. Can't infer sign of delta from sinusoid,
        # therefore just pick one of the possible values
        if model_parameters["w_0"].fixed_to is not None:
            model_parameters["w"].heuristic = model_parameters["w_0"].fixed_to + delta
        elif model_parameters["w"].fixed_to is not None:
            model_parameters["w_0"].heuristic = model_parameters["w"].fixed_to - delta
        else:
            model_parameters["w_0"].fixed_to = 0.0
            model_parameters["w"].heuristic = delta
