from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_power, scale_x, scale_y
from . import heuristics
from .heuristics import get_spectrum


class Gaussian(Model):
    """Gaussian model according to::

        y = a / (sigma * sqrt(2*pi)) * exp(-0.5*((x-x0)/(sigma))^2) + y0

    See :meth:`_func` for parameter details.
    """

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [True]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        x0: ModelParameter(scale_x()),
        y0: ModelParameter(scale_y()),
        a: ModelParameter(scale_func=scale_power(x_power=1, y_power=1)),
        sigma: ModelParameter(lower_bound=0, scale_func=scale_x()),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param y0: y-axis offset
        :param a: y-axis scale factor. The Gaussian is normalized such that its integral
          is equal to ``a``.
        :param sigma: distribution half-width at ``1/e`` of maximum height is
          ``2*sigma`` (``sigma`` is the ``1/sqrt(e)`` radius).
        """
        y = (
            a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
            + y0
        )
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TX):
        x = np.squeeze(x)
        y = np.squeeze(y)

        basic_heuristics = self._estimate_parameters_basic(x=x, y=y)
        try:
            fft_heuristics = self._estimate_params_fft(x=x, y=y)
        except Exception:
            fft_heuristics = basic_heuristics
        try:
            peak_heuristics = self._estimate_params_peak(x=x, y=y)
        except Exception:
            peak_heuristics = basic_heuristics

        cost_fft = np.sum((y - self.func(x, fft_heuristics)) ** 2)
        cost_peak_heuristcs = np.sum((y - self.func(x, peak_heuristics)) ** 2)

        best = fft_heuristics if cost_fft < cost_peak_heuristcs else peak_heuristics

        for param_name, heuristic in best.items():
            self.parameters[param_name].heuristic = heuristic

    def _estimate_parameters_basic(self, x: TX, y: TY) -> Dict[str, float]:
        # fallback heuristics (e.g. don't fail even if we have a dataset that's too
        # small for the better heuristics)
        x0 = self.parameters["x0"].get_initial_value(np.mean(x))
        y0 = self.parameters["y0"].get_initial_value(0.5 * (y[0] + y[-1]))
        sigma = self.parameters["sigma"].get_initial_value(np.ptp(x) / 2)

        x0_ind = np.argmin(np.abs(x - x0))
        peak = y[x0_ind] - y0
        a = peak * sigma * np.sqrt(2 * np.pi)
        a = self.parameters["a"].get_initial_value(a)

        return {"a": a, "sigma": sigma, "y0": y0, "x0": x0}

    def _estimate_params_peak(self, x: TX, y: TY) -> Dict[str, float]:
        x0 = heuristics.get_sym_x(x=x, y=y)
        x0_ind = np.argmin(np.abs(x - x0))
        x0 = self.parameters["x0"].get_initial_value(x0)

        y0 = y[np.argmax(np.abs(x - x0))]
        y0 = self.parameters["y0"].get_initial_value(y0)

        peak = y[x0_ind] - y0
        inside = np.abs(y - y[x0_ind]) <= np.abs(peak * (1 - np.exp(-1)))
        full_width_1_e = np.ptp(x[inside])

        sigma = full_width_1_e / (2 * np.sqrt(2))
        sigma = self.parameters["sigma"].get_initial_value(sigma)

        a = peak * sigma * np.sqrt(2 * np.pi)
        a = self.parameters["a"].get_initial_value(a)

        return {"a": a, "sigma": sigma, "y0": y0, "x0": x0}

    def _estimate_params_fft(self, x: TX, y: TY) -> Dict[str, float]:
        # Gaussian Fourier Transform:
        #   F[A * exp(-(x/w)^2)](k) = A * sqrt(pi) * w * exp(-(pi*k*w)^2)
        #
        # This heuristic generally works extremely well when we have enough data but
        # struggles for smaller datasets

        fft_heuristics = {}
        y0 = self.parameters["y0"].get_initial_value(np.mean([y[0], y[-1]]))
        fft_heuristics["y0"] = y0

        omega, spectrum = get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        def fun(omega, alpha, w):
            return alpha * np.exp(-((0.5 * omega * w) ** 2))

        alpha0 = abs_spectrum[0]
        w0 = 2 / omega[abs_spectrum < alpha0 * np.exp(-1)][0]
        (alpha, w), _ = curve_fit(
            f=fun, xdata=omega, ydata=abs_spectrum, p0=[alpha0, w0]
        )

        peak_idx = np.argmax(np.abs(y - y0))
        y_peak = y[peak_idx]
        sgn = 1 if y_peak > y0 else -1

        fft_heuristics["sigma"] = w / np.sqrt(2)
        fft_heuristics["a"] = sgn * alpha

        x0 = heuristics.find_x_offset_sym_peak_fft(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=4 / w,
            test_pts=x[peak_idx],
            defaults=fft_heuristics,
        )

        fft_heuristics["x0"] = x0

        return fft_heuristics

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Derived parameters:

        * ``FWHMH``: full width at half-maximum height
        * ``peak``: peak height above ``y0``
        * ``w0``: full width at ``1/e`` max height. For Gaussian beams this is the beam
          waist

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
        ) ** 2 + (fit_uncertainties["sigma"] * a / (sigma**2 * np.sqrt(2 * np.pi))) ** 2
        derived_uncertainties["w0"] = 4 * fit_uncertainties["sigma"]

        return derived_params, derived_uncertainties
