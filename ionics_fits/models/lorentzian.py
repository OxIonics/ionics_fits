from typing import Dict, TYPE_CHECKING
import numpy as np

from .exponential import Exponential
from .utils import get_spectrum
from .. import NormalFitter, Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class Lorentzian(Model):
    """Lorentzian model according to:
    y = a * fwhmh^2 / ((x - x0)^2 + fwhmh^2) + y0

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis offset
      - a: peak value of the function above y0
      - fwhmh: full width at half maximum height of the function

    Derived parameters:
        None
    """

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 1

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        a: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        fwhmh: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: x_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        y = a * fwhmh**2 / ((x - x0) ** 2 + fwhmh**2) + y0
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

        Heuristic results should be stored in :param model_parameters: using the
        `ModelParameter`'s `heuristic` attribute. This ensures that all information
        passed in by the user (fixed values, initial values, bounds) is used correctly.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        omega, spectrum = get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        model = Exponential()
        model.parameters["y0"].heuristic = np.max(abs_spectrum)
        model.parameters["y_inf"].heuristic = 0.0
        fit = NormalFitter(omega, abs_spectrum, model)

        model_parameters["y0"].heuristic = np.mean([y[0], y[-1]])
        y0 = model_parameters["y0"].get_initial_value()

        peak_idx = np.argmax(np.abs(y - y0))
        y_peak = y[peak_idx]
        sgn = 1 if y_peak > y0 else -1

        model_parameters["fwhmh"].heuristic = 1 / fit.values["tau"]
        fwhmh = model_parameters["fwhmh"].get_initial_value()
        model_parameters["a"].heuristic = fit.values["y0"] * sgn * 2 / fwhmh

        cut_off = 2 * fit.values["tau"]

        x0 = self.find_x_offset_sym_peak(
            x=x,
            y=y,
            parameters=model_parameters,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=cut_off,
            test_pts=x[peak_idx],
        )

        model_parameters["x0"].heuristic = x0
