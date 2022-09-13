from typing import Dict, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
from ..utils import Array
import ionics_fits as fits


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

    # pytype: disable=attribute-error
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

    # pytype: enable=attribute-error

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
        omega, spectrum = fits.models.utils.get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        model = fits.models.Exponential()
        model.parameters["y0"].initialise(np.max(abs_spectrum))
        model.parameters["y_inf"].initialise(0)
        fit = fits.NormalFitter(omega, abs_spectrum, model)

        y0_guess = np.mean([y[0], y[-1]])
        peak_guess = y[np.argmax(np.abs(y - y0_guess))]
        sgn = 1 if peak_guess > y0_guess else -1

        model_parameters["y0"].initialise(y0_guess)
        fwhmh = model_parameters["fwhmh"].initialise(1 / fit.values["tau"])
        model_parameters["a"].initialise(fit.values["y0"] * sgn * 2 / fwhmh)

        cut_off = 2 * fit.values["tau"]
        model_parameters["x0"].initialise(
            self.find_x_offset_fft(x, omega, spectrum, cut_off)
        )
