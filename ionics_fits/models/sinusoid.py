import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

from . import heuristics, ReparametrizedModel
from .. import common, Model, ModelParameter
from ..utils import Array
from . import utils

if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class Sinusoid(Model):
    """Generalised sinusoid fit according to:
    y = Gamma * a * sin[omega * (x - x0) + phi] + y0
    where Gamma = exp(-x / tau).

    Fit parameters (all floated by default unless stated otherwise):
      - a: initial (x = 0) amplitude of the sinusoid
      - omega: angular frequency
      - phi: phase offset (radians)
      - y0: y-axis offset
      - x0: x-axis offset (fixed to 0 by default)
      - tau: decay/growth constant (fixed to np.inf by default)

    Derived parameters:
      - f: frequency
      - phi_cosine: cosine phase (phi + pi/2)
      - contrast: peak-to-peak amplitude of the pure sinusoid
      - min/max: min / max values of the pure sinusoid
      - period: period of oscillation
      - TODO: peak values of the damped sinusoid as well as `x` value that the peak
          occurs at.

    All phases are in radians, frequencies are in angular units.

    x0 and phi0 are equivalent parametrisations for the phase offset, but in some cases
    it works out convenient to have access to both (e.g. one as a fixed offset, the
    other floated). At most one of them should be floated at once. By default, x0 is
    fixed at 0 and phi0 is floated.
    """

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[bool, bool]:
        return True, True

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        a: ModelParameter(lower_bound=0, scale_func=common.scale_y),
        omega: ModelParameter(lower_bound=0, scale_func=common.scale_x_inv),
        phi: utils.PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
            scale_func=common.scale_invariant,
        ),
        y0: ModelParameter(scale_func=common.scale_y),
        x0: ModelParameter(fixed_to=0, scale_func=common.scale_x),
        tau: ModelParameter(
            lower_bound=0,
            fixed_to=np.inf,
            scale_func=common.scale_x,
        ),
    ) -> Array[("num_samples",), np.float64]:
        Gamma = np.exp(-x / tau)
        y = Gamma * a * np.sin(omega * (x - x0) + phi) + y0
        return y

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        # We don't have good heuristics for these parameters
        self.parameters["y0"].heuristic = np.mean(y)
        self.parameters["tau"].heuristic = np.max(x)

        omega, spectrum = heuristics.get_spectrum(
            x, y, density_units=False, trim_dc=True
        )
        spectrum = np.abs(spectrum)
        peak = np.argmax(spectrum)

        self.parameters["a"].heuristic = spectrum[peak]
        self.parameters["omega"].heuristic = omega[peak]

        phi, _ = heuristics.param_min_sqrs(
            model=self,
            x=x,
            y=y,
            scanned_param="phi",
            scanned_param_values=np.linspace(-np.pi, np.pi, num=20),
            defaults={"x0": 0},
        )

        phi = self.parameters["phi"].clip(phi)

        if self.parameters["x0"].fixed_to is None:
            if self.parameters["phi"].fixed_to is None:
                raise ValueError("Only one of 'x0' and 'phi' may be floated at once")

            self.parameters["phi"].heuristic = 0
            self.parameters["x0"].heuristic = (
                -phi / self.parameters["omega"].get_initial_value()
            )
        else:
            self.parameters["phi"].heuristic = phi
            self.parameters["x0"].heuristic = 0.0

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_params["f"] = fitted_params["omega"] / (2 * np.pi)
        derived_params["phi_cosine"] = fitted_params["phi"] + np.pi / 2
        derived_params["contrast"] = 2 * np.abs(fitted_params["a"])
        derived_params["min"] = fitted_params["y0"] - np.abs(fitted_params["a"])
        derived_params["max"] = fitted_params["y0"] + np.abs(fitted_params["a"])
        derived_params["period"] = 2 * np.pi / fitted_params["omega"]

        derived_uncertainties = {}
        derived_uncertainties["f"] = fit_uncertainties["omega"] / (2 * np.pi)
        derived_uncertainties["phi_cosine"] = fit_uncertainties["phi"]
        derived_uncertainties["contrast"] = 2 * fit_uncertainties["a"]
        derived_uncertainties["min"] = np.sqrt(
            fit_uncertainties["y0"] ** 2 + fit_uncertainties["a"] ** 2
        )
        derived_uncertainties["max"] = np.sqrt(
            fit_uncertainties["y0"] ** 2 + fit_uncertainties["a"] ** 2
        )
        derived_uncertainties["period"] = (
            2 * np.pi * fit_uncertainties["omega"] / (fitted_params["omega"] ** 2)
        )

        return derived_params, derived_uncertainties


class SineMinMax(ReparametrizedModel):
    """Sinusoid parametrised by minimum / maximum values instead of offset / amplitude.

    This class is equivalent to :class Sinusoid: except that the `a` and `y0` parameters
    are replaced with new `min` and `max` parameters defined by:
      - `min = y0 - a`
      - `max = y0 + a`
    """

    def __init__(self):
        super().__init__(
            model=Sinusoid(),
            new_params={
                "min": ModelParameter(scale_func=common.scale_y),
                "max": ModelParameter(scale_func=common.scale_y),
            },
            bound_params=["a", "y0"],
        )

    @staticmethod
    def bound_param_values(param_values: Dict[str, float]) -> Dict[str, float]:
        return {
            "a": 0.5 * (param_values["max"] - param_values["min"]),
            "y0": 0.5 * (param_values["max"] + param_values["min"]),
        }

    @staticmethod
    def bound_param_uncertainties(
        param_values: Dict[str, float], param_uncertainties: Dict[str, float]
    ) -> Dict[str, float]:
        err = 0.5 * np.sqrt(
            param_uncertainties["max"] ** 2 + param_uncertainties["min"] ** 2
        )
        return {"a": err, "y0": err}

    @staticmethod
    def new_param_values(model_param_values: Dict[str, float]) -> Dict[str, float]:
        return {
            "max": (model_param_values["y0"] + model_param_values["a"]),
            "min": (model_param_values["y0"] - model_param_values["a"]),
        }
