from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
from ..utils import Array
import ionics_fits as fits

if TYPE_CHECKING:
    num_samples = float


class RabiFlop(Model):
    """Base class for time-domain / frequency-domain exponentially damped Rabi flop fits
    according to:
        y = P0 * y0 + P1 * y1

        where:
        - P1 = 1 - P0
        - y1 = 1 - y0
        - y0 = Gamma * (contrast * (omega * t / 2 * sinc(W*t/2))^2 + P_lower - c) + c
        - Gamma = exp(-t/tau)
        - t = max(0, t_pulse - t_dead)
        - contrast = P_upper - P_lower
        - c = 0.5 * (P_upper + P_lower)
        - W = sqrt(omega^2 + (detuning)^2)

    This class is not intended to be instantiated directly, use one of the
    :class RabiFlopFreq: or :class RabiFlopTime: sub-classes instead.

    For frequency scans, we set:
      - detuning = x + delta
      - t = t_pulse - t_dead

    For time scans, we set:
      - detuning = delta
      - t = max(x - t_dead, 0)

    Fit parameters (all floated by default unless stated otherwise):
        - P1: initial upper-state population (fixed to 1 by default)
        - P_upper: upper readout level (fixed to 1 by default)
        - P_lower: lower readout level (fixed to 0 by default)
        - delta: the detuning offset in angular units
        - omega: Rabi frequency
        - t_pulse: pulse duration (detuning scans only). For pulse areas >> 1 this
          should either be fixed or have a user-supplied value.
        - t_dead: dead_time (fixed to 0 by default)
        - tau: decay time constant (fixed to np.inf by default)

    Derived parameters:
        - t_pi: pi-time including dead-time (so t_2pi != 2*t_pi). NB this is not the
          time for maximum population transfer for finite tau (we can add that as a
          separate derived parameter if it proves useful)
        - t_pi_2: pi/2-time including dead-time (so t_pi != 2*t_pi_2)

    All phases are in radians, detunings are in angular units.
    """

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
        P1: ModelParameter(
            lower_bound=0,
            upper_bound=1,
            fixed_to=1,
        ),
        P_upper: ModelParameter(
            lower_bound=0,
            upper_bound=1,
            fixed_to=1,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        P_lower: ModelParameter(
            lower_bound=0,
            upper_bound=1,
            fixed_to=0,
            scale_func=lambda x_scale, y_scale, _: y_scale,
        ),
        delta: ModelParameter(),
        omega: ModelParameter(lower_bound=0),
        t_dead: ModelParameter(
            lower_bound=0,
            fixed_to=0,
        ),
        tau: ModelParameter(
            lower_bound=0,
            fixed_to=np.inf,
        ),
        t_pulse: ModelParameter(lower_bound=0) = None,
    ) -> Array[("num_samples",), np.float64]:
        """
        :param x: tuple of t, delta
        """
        t = np.clip(x[0], a_min=0, a_max=None)
        detuning = x[1]

        contrast = P_upper - P_lower
        c = 0.5 * (P_upper + P_lower)

        Gamma = np.exp(-t / tau)
        W = np.sqrt(np.power(omega, 2) + np.power(detuning, 2))

        # NB np.sinc(x) = sin(pi*x)/(pi*x)
        y0 = (
            Gamma
            * (
                contrast * np.power((omega * t / 2 * np.sinc(W * t / (2 * np.pi))), 2)
                + P_lower
                - c
            )
            + c
        )

        P0 = 1 - P1
        y1 = 1 - y0

        y = P0 * y0 + P1 * y1
        return y

    # pytype: enable=invalid-annotation

    @staticmethod
    def calculate_derived_params(
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float]
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
        t_pi = np.pi / omega + fitted_params["t_dead"]
        t_pi_2 = np.pi / (2 * omega) + fitted_params["t_dead"]

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
    def __init__(self):
        super().__init__()

        delta = self.parameters["delta"]
        omega = self.parameters["omega"]
        t_pulse = self.parameters["t_pulse"]
        t_dead = self.parameters["t_dead"]
        tau = self.parameters["tau"]

        delta.scale_func = lambda x_scale, y_scale, _: x_scale
        omega.scale_func = lambda x_scale, y_scale, _: x_scale
        t_pulse.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        t_dead.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        tau.scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        t = param_values["t_pulse"] - param_values["t_dead"]
        detuning = x + param_values["delta"]
        return super().func(
            (t, detuning), param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Sets initial values for model parameters based on heuristics. Typically
        called during `Fitter.fit`.

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
        model_parameters["P1"].initialise(1 if y[0] > 0.5 else 0)
        model_parameters["P_upper"].initialise(1)
        model_parameters["P_lower"].initialise(0)
        model_parameters["t_dead"].initialise(0)
        model_parameters["tau"].initialise(np.inf)

        # there isn't a simple analytic form for the Fourier transform of a Rabi
        # flop in the general case. However in the low pulse area limit (and
        # ignoring decay etc) the Rabi flop function tends to the sinc^2 function:
        #   (omega * t_pulse / 2) ^2 * sinc(delta*t_pulse/2)
        # This heuristic breaks down when: omega * t_pulse ~ pi
        model = fits.models.Sinc2()
        model.parameters["y0"].fixed_to = 1
        fit = fits.NormalFitter(x, y, model)

        a = np.abs(fit.values["a"])
        t_pulse = model_parameters["t_pulse"].initialise(fit.values["w"] * 2)
        model_parameters["omega"].initialise(2 * np.sqrt(a) / t_pulse)
        model_parameters["delta"].initialise(-fit.values["x0"])


class RabiFlopTime(RabiFlop):
    def __init__(self):
        super().__init__()

        del self.parameters["t_pulse"]

        delta = self.parameters["delta"]
        omega = self.parameters["omega"]
        t_dead = self.parameters["t_dead"]
        tau = self.parameters["tau"]

        delta.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        delta.lower_bound = 0
        omega.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        t_dead.scale_func = lambda x_scale, y_scale, _: x_scale
        tau.scale_func = lambda x_scale, y_scale, _: 1 / x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        t = x - param_values["t_dead"]
        detuning = param_values["delta"]
        return super().func(
            (t, detuning), param_values
        )  # pytype: disable=wrong-arg-types

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Sets initial values for model parameters based on heuristics. Typically
        called during `Fitter.fit`.

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
        # Could back P1 out from the phase of the sinusoid, but keeping it simple
        # for now...
        P1 = model_parameters["P1"].initialise(1 if y[0] > 0.5 else 0)
        model_parameters["P_upper"].initialise(1)
        model_parameters["P_lower"].initialise(0)
        model_parameters["t_dead"].initialise(0)
        model_parameters["tau"].initialise(np.inf)

        model = fits.models.Sinusoid()
        model.parameters["phi"].fixed_to = np.pi / 2 if P1 == 1 else 0
        fit = fits.NormalFitter(x, y, model)

        # (omega / W) ^ 2 * sin(0.5 * W * t) ^ 2
        # = 0.5 * (omega / W) ^ 2 * (1 - cos(W * t))
        W = fit.values["omega"]
        omega = np.sqrt(2 * fit.values["a"]) * W
        # avoid divide by zero errors from numerical noise when delta ~= 0
        delta = 0 if omega >= W else np.sqrt(np.power(W, 2) - np.power(omega, 2))

        model_parameters["omega"].initialise(omega)
        model_parameters["delta"].initialise(delta)
