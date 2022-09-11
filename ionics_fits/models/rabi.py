from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
from ..utils import Array
import ionics_fits as fits

if TYPE_CHECKING:
    num_samples = float

# TODO: derived params

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
        - W = sqrt(omega^2 + (delta + detuning_offset)^2)

    NB scanning delta or t_pulse

    Fit parameters (all floated by default unless stated otherwise):
        - P1: initial upper-state population (fixed to 1 by default)
        - P_upper: upper readout level (fixed to 1 by default)
        - P_lower: lower readout level (fixed to 0 by default)
        - detuning_offset: the detuning offset in angular units
        - omega: Rabi frequency
        - t_pulse: pulse duration (detuning scans only). For pulse areas >> 1 this
          should either be fixed or have a user-supplied value.
        - t_dead: dead_time (fixed to 0 by default)
        - tau: decay time constant (fixed to np.inf by default)

    Derived parameters:
        - t_pi: pi-time including dead-time (so t_2pi != 2*t_pi), is not the time for
          maximum population transfer for finite tau (TODO: add that as a derived
          parameter!)
        - t_pi_2: pi/2-time including dead-time (so t_pi != 2*t_pi_2)
        - TODO: do we want pulse area error, etc?

    All phases are in radians, detunings are in angular units.
    """

    def _func(
        self,
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
        detuning_offset: ModelParameter(),
        omega: ModelParameter(lower_bound=0),
        t_pulse: ModelParameter(lower_bound=0),
        t_dead: ModelParameter(
            lower_bound=0,
            fixed_to=0,
        ),
        tau: ModelParameter(
            lower_bound=0,
            fixed_to=np.inf,
        ),
    ) -> Array[("num_samples",), np.float64]:
        """
        :param x: tuple of t, delta
        """
        t = np.clip(x[0], a_min=0, a_max=None)
        delta = x[1]

        contrast = P_upper - P_lower
        c = 0.5 * (P_upper + P_lower)

        Gamma = np.exp(-t / tau)
        W = np.sqrt(np.power(omega, 2) + np.power(delta, 2))

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


class RabiFlopFreq(RabiFlop):
    def __init__(self):
        super().__init__()

        detuning_offset = self.parameters["detuning_offset"]
        omega = self.parameters["omega"]
        t_pulse = self.parameters["t_pulse"]
        t_dead = self.parameters["t_dead"]
        tau = self.parameters["tau"]

        detuning_offset.scale_func = lambda x_scale, y_scale, _: x_scale
        omega.scale_func = lambda x_scale, y_scale, _: x_scale
        t_pulse.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        t_dead.scale_func = lambda x_scale, y_scale, _: 1 / x_scale
        tau.scale_func = lambda x_scale, y_scale, _: x_scale

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        t = param_values["t_pulse"] - param_values["t_dead"]
        delta = x + param_values["detuning_offset"]
        return super().func((t, delta), param_values)

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

        if (
            model_parameters["t_pulse"].get_initial_value() is None
            or model_parameters["omega"].get_initial_value() is None
        ):
            # there isn't a simple analytic form for the Fourier transform of a Rabi
            # flop in the general case, but in the low pulse area limit it tends to a
            # triangle function (transform of sinc^2)
            #
            # This heuristic breaks down when: omega * t_pulse ~ pi
            pgram_omega, pgram = fits.models.utils.get_pgram(x, y)

            tri = fits.models.triangle.Triangle()
            tri.parameters["x0"].fixed_to = 0
            tri.parameters["y0"].initialise(max(pgram))
            tri.parameters["sym"].fixed_to = 0
            tri.parameters["y_min"].fixed_to = 0

            fit = fits.NormalFitter(pgram_omega, pgram, model=tri)
            intercept = fit.values["y0"] / -fit.values["k"]

            model_parameters["t_pulse"].initialise(intercept)
            model_parameters["omega"].initialise(fit.values["y0"] / 2)

        if model_parameters["detuning_offset"].get_initial_value() is None:
            w = 2 * np.pi / model_parameters["t_pulse"].get_initial_value()
            model_parameters["detuning_offset"].initialise(
                self.find_x_offset(x, y, model_parameters, w, "detuning_offset")
            )
