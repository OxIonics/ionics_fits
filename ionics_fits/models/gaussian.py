from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .utils import get_spectrum
from .. import common, Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class Gaussian(Model):
    """Gaussian model according to:
    y = a / (sigma * sqrt(2*pi)) * exp(-0.5*((x-x0)/(sigma))^2) + y0

    Fit parameters (all floated by default unless stated otherwise):
      - x0: x-axis offset
      - y0: y-axis offset
      - a: y-axis scale factor. The Gaussian is normalized such that its integral is
        equal to a.
      - sigma: distribution half-width at 1/e of maximum height is 2*sigma (sigma is the
        1/sqrt(e) radius).

    Derived parameters:
      - FWHMH: full width at half-maximum height
      - peak: peak height above y0
      - w0: full width at 1/e max height. For Gaussian beams this is the beam waist
    """

    def get_num_y_channels(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[bool, bool]:
        return True, True

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=common.scale_x),
        y0: ModelParameter(scale_func=common.scale_y),
        a: ModelParameter(scale_func=common.scale_power(x_power=1, y_power=1)),
        sigma: ModelParameter(lower_bound=0, scale_func=common.scale_x),
    ) -> Array[("num_samples",), np.float64]:
        y = (
            a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
            + y0
        )
        return y

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
        y = np.squeeze(y)

        # Gaussian Fourier Transform:
        #   F[A * exp(-(x/w)^2)](k) = A * sqrt(pi) * w * exp(-(pi*k*w)^2)
        #
        # Half-width at 1/e when k = 1/(pi*w)
        omega, spectrum = get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        k = omega / (2 * np.pi)

        peak = np.max(abs_spectrum)
        W = peak / np.exp(1)
        width = 1 / (np.pi * k[np.argmin(np.abs(abs_spectrum - W))])

        sigma = width / 2
        a = peak * np.pi * np.sqrt(2)

        self.parameters["y0"].heuristic = np.mean([y[0], y[-1]])
        y0 = self.parameters["y0"].get_initial_value()
        peak_idx = np.argmax(np.abs(y - y0))
        y_peak = y[peak_idx]
        sgn = 1 if y_peak > y0 else -1

        self.parameters["a"].heuristic = a * sgn
        self.parameters["sigma"].heuristic = sigma

        cut_off = 2 * omega[np.argmin(np.abs(abs_spectrum - W))]

        x0 = self.find_x_offset_sym_peak(
            x=x,
            y=y,
            parameters=self.parameters,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=cut_off,
            test_pts=x[peak_idx],
        )

        self.parameters["x0"].heuristic = x0

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        sigma = fitted_params["sigma"]
        a = fitted_params["a"]

        derived_params = {}
        derived_params["FWHMH"] = 2.35482 * sigma
        derived_params["peak"] = a / (sigma * np.sqrt(2 * np.pi))
        derived_params["w0"] = 4 * sigma

        derived_uncertainties = {}
        derived_uncertainties["FWHMH"] = 2.35482 * fit_uncertainties["sigma"]
        derived_uncertainties["peak"] = (
            fit_uncertainties["a"] / (sigma * np.sqrt(2 * np.pi))
        ) ** 2 + (
            fit_uncertainties["sigma"] * a / (sigma**2 * np.sqrt(2 * np.pi))
        ) ** 2
        derived_uncertainties["w0"] = 4 * fit_uncertainties["sigma"]

        return derived_params, derived_uncertainties
