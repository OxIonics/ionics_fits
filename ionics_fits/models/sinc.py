from typing import Dict, TYPE_CHECKING
import numpy as np

from .. import NormalFitter, Model, ModelParameter
from ..utils import Array
import ionics_fits as fits

if TYPE_CHECKING:
    num_samples = float


class Sinc(Model):
    """Sinc function according to:
    y = a * sin(w * (x - x0)) / (w * (x - x0)) + y0

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis offset
      - a: amplitude
      - w: x scale factor

    Derived parameters:
      None
    """

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=lambda x_scale, y_scale, _: x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        a: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        w: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: 1 / x_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * np.sinc(x) + y0
        return y

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
        y0 = model_parameters["y0"].initialise(np.mean([y[0], y[-1]]))

        omega, spectrum = fits.models.utils.get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        # Fourier transform of a sinc is a rectangle
        rect = fits.models.rectangle.Rectangle()
        rect.parameters["y0"].fixed_to = 0
        rect.parameters["x_l"].fixed_to = 0
        rect.parameters["a"].initialise(max(abs_spectrum))

        fit = NormalFitter(omega, abs_spectrum, model=rect)

        w = model_parameters["w"].initialise(fit.values["x_r"])
        sgn = 1 if y[np.argmax(np.abs(y - y0))] > y0 else -1
        model_parameters["a"].initialise(2 * w * fit.values["a"] * sgn)
        model_parameters["x0"].initialise(self.find_x_offset_fft(x, omega, spectrum, w))


class Sinc2(Model):
    """Sinc-squared function according to:
    y = a * (sin(w * (x - x0)) / (w * (x - x0)))^2 + y0

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis offset
      - a: amplitude
      - w: x scale factor

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
        w: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: 1 / x_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * np.power(np.sinc(x), 2) + y0
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
        y0 = model_parameters["y0"].initialise(np.mean([y[0], y[-1]]))

        omega, spectrum = fits.models.utils.get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        # Fourier transform of a sinc^2 is a triangle function
        tri = fits.models.triangle.Triangle()
        tri.parameters["x0"].fixed_to = 0
        tri.parameters["y0"].initialise(max(abs_spectrum))
        tri.parameters["sym"].fixed_to = 0
        tri.parameters["y_min"].fixed_to = 0

        fit = NormalFitter(omega, abs_spectrum, model=tri)

        intercept = fit.values["y0"] / -fit.values["k"]
        sgn = 1 if y[np.argmax(np.abs(y - y0))] > y0 else -1

        model_parameters["w"].initialise(0.5 * intercept)
        model_parameters["a"].initialise(fit.values["y0"] * sgn * intercept)

        model_parameters["x0"].initialise(
            self.find_x_offset_fft(x, omega, spectrum, intercept)
        )
