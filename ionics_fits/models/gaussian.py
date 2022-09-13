from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

from .. import Model, ModelParameter
from ..utils import Array
import ionics_fits as fits


if TYPE_CHECKING:
    num_samples = float


class Gaussian(Model):
    """Gaussian model according to:
    y = a / (sigma * sqrt(2*pi)) * exp(-(x-x0)^2/(2*sigma)^2) + y0

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

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
        x0: ModelParameter(scale_func=lambda x_scale, y_scale, _: None),  # x_scale),
        y0: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        a: ModelParameter(scale_func=lambda x_scale, y_scale, _: y_scale),
        sigma: ModelParameter(
            lower_bound=0, scale_func=lambda x_scale, y_scale, _: x_scale
        ),
    ) -> Array[("num_samples",), np.float64]:
        y = (
            a
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-np.power((x - x0) / (2 * sigma), 2))
            + y0
        )
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
        # Gaussian Fourier Transform:
        #   F[A * exp(-(x/w)^2)](k) = A * sqrt(pi) * w * exp(-(pi*k*w)^2)
        #
        # Half-width at 1/e when k = 1/(pi*w)
        # Peak value: A * sqrt(pi) * w = A * 2 * sqrt(pi) * sigma
        #   where A = a / (sigma * np.sqrt(2*np.pi))
        #   so peak is: a / (sigma * np.sqrt(2*np.pi)) * 2 * sqrt(pi) * sigma
        #              = a * sqrt(2)
        omega, pgram = fits.models.utils.get_pgram(x, y - np.mean(y))
        k = omega / (2 * np.pi)

        peak = np.max(pgram)
        W = peak / np.exp(1)
        width = 1 / (np.pi * k[np.argmin(np.abs(pgram - W))])

        # NB this (empirical) expression for a is not what I was expecting (factor of 2)
        # I ran out of steam before tracking down where I went wrong...ahem...I mean
        # "I left this as an exercise for the reader". Normalization of the pgram?
        sigma = width / 2
        a = peak * 2 * np.sqrt(2)

        y0_guess = np.mean([y[0], y[-1]])
        peak_guess = y[np.argmax(np.abs(y - y0_guess))]
        sgn = 1 if peak_guess > y0_guess else -1

        model_parameters["a"].initialise(a * sgn)
        model_parameters["sigma"].initialise(sigma)
        model_parameters["y0"].initialise(y0_guess)

        model_parameters["x0"].initialise(
            self.find_x_offset(x, y, model_parameters, sigma)
        )

    @staticmethod
    def calculate_derived_params(
        fitted_params: Dict[str, float], fit_uncertainties: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
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