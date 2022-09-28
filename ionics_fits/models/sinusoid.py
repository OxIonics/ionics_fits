import copy
import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

from .. import Model, ModelParameter
from ..utils import Array
from . import utils

if TYPE_CHECKING:
    num_samples = float


class Sinusoid(Model):
    """Generalised sinusoid fit according to:
    y = Gamma*a*sin(omega*(x - x0) + phi) + y0
    where Gamma = exp(-x/tau)

    Fit parameters (all floated by default unless stated otherwise):
      - a: initial (x = 0) amplitude of the decaying sinusoid
      - omega: angular frequency
      - phi: phase offset (radians)
      - y0: y-axis offset
      - x0: x-axis offset (fixed to 0 by default)
      - tau: decay constant (fixed to np.inf by default)

    Derived parameters:
      - f: frequency
      - phi_cosine: cosine phase (phi + pi/2)
      - min/max: min / max values of the undamped sinusoid (including the offset and
          decay).
      - period: period of oscillation
      - TODO: peak values of the damped sinusoid as well as `x` value that the peak
          occurs at.

    All phases are in radians, frequencies are in angular units.

    x0 and phi0 are equivalent parametrisations for the phase offset, but in some cases
    it works out convenient to have access to both (e.g. one as a fixed offset, the
    other floated). At most one of them should be floated at once. By default, x0 is
    fixed at 0 and phi0 is floated.
    """

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        a: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: y_scale
        ),
        omega: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: 1 / x_scale
        ),
        phi: utils.PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
        ),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        x0: ModelParameter(fixed_to=0, scale_func=lambda x_scale, y_scale, _: x_scale),
        tau: ModelParameter(
            fixed_to=np.inf, scale_func=lambda x_scale, y_scale, _: x_scale
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
        # We don't have good heuristics for these parameters
        model_parameters["y0"].initialise(np.mean(y))
        model_parameters["tau"].initialise(np.max(x))

        omega, spectrum = utils.get_spectrum(x, y, density_units=False, trim_dc=True)
        spectrum = np.abs(spectrum)
        peak = np.argmax(spectrum)

        model_parameters["a"].initialise(spectrum[peak] * 2)
        model_parameters["omega"].initialise(omega[peak])

        phi_params = copy.deepcopy(model_parameters)
        phi_params["x0"].initialise(0)
        phi, _ = self.param_min_sqrs(
            x=x,
            y=y,
            parameters=phi_params,
            scanned_param="phi",
            scanned_param_values=np.linspace(-np.pi, np.pi, num=20),
        )
        phi = model_parameters["phi"].clip(phi)

        if model_parameters["x0"].fixed_to is None:
            if model_parameters["phi"].fixed_to is None:
                raise ValueError("Only one of 'x0' and 'phi' may be floated at once")

            model_parameters["phi"].initialise(0)
            model_parameters["x0"].initialise(
                -phi / model_parameters["omega"].get_initial_value()
            )
        else:
            model_parameters["phi"].initialise(phi)
            model_parameters["x0"].initialise(0)

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
        derived_params = {}
        derived_params["f"] = fitted_params["omega"] / (2 * np.pi)
        derived_params["phi_cosine"] = fitted_params["phi"] + np.pi / 2
        derived_params["min"] = fitted_params["y0"] - np.abs(fitted_params["a"])
        derived_params["max"] = fitted_params["y0"] + np.abs(fitted_params["a"])
        derived_params["period"] = 2 * np.pi / fitted_params["omega"]

        derived_uncertainties = {}
        derived_uncertainties["f"] = fit_uncertainties["omega"] / (2 * np.pi)
        derived_uncertainties["phi_cosine"] = fit_uncertainties["phi"]
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
