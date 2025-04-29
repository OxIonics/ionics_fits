from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_x, scale_y
from . import heuristics
from .heuristics import get_spectrum


class Lorentzian(Model):
    """Lorentzian model according to::

        y = a * (0.5 * fwhmh)^2 / ((x - x0)^2 + (0. 5 * fwhmh)^2) + y0

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
        x0: ModelParameter(scale_func=scale_x()),
        y0: ModelParameter(scale_func=scale_y()),
        a: ModelParameter(scale_func=scale_y()),
        fwhmh: ModelParameter(lower_bound=0, scale_func=scale_x()),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param y0: y-axis offset
        :param a: peak value of the function above ``y0``
        :param fwhmh: full width at half maximum height of the function
        """
        y = a * (0.5 * fwhmh) ** 2 / ((x - x0) ** 2 + (0.5 * fwhmh) ** 2) + y0
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        # Fourier transform:
        #  f(y) = a * (0.5 * fwhmh)^2 / ((x - x0)^2 + (0.5 * fwhmh)^2) + y0
        #  f(k) = a * 0.5 * fwhmh * pi * exp(-2*pi*i*k*x0) * exp(-2*pi*0.5*fwhmh*|k|)
        omega, spectrum = get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)
        k = omega / (2 * np.pi)

        peak = abs_spectrum[0]
        W = peak * np.exp(-1)

        idx_1_e = np.argmin(np.abs(abs_spectrum - W))

        # We usually don't have great spectral resolution around the peak so interpolate
        if k[idx_1_e] > W:
            upper_idx = idx_1_e
            lower_idx = idx_1_e + 1
        else:
            upper_idx = idx_1_e - 1
            lower_idx = idx_1_e

        # if we don't have enough data to figure this out, set the half-width to one
        # sample wide
        df_dk = (abs_spectrum[lower_idx] - abs_spectrum[upper_idx]) / (k[1] - k[0])
        if df_dk == 0:
            tau = k[1] - k[0]
        else:
            df = abs_spectrum[idx_1_e] - W
            tau = k[idx_1_e] - df / df_dk

        fwhmh = 1 / (np.pi * tau)
        a = 2 * peak / (np.pi * fwhmh)

        self.parameters["y0"].heuristic = np.mean([y[0], y[-1]])
        y0 = self.parameters["y0"].get_initial_value()
        peak_idx = np.argmax(np.abs(y - y0))
        y_peak = y[peak_idx]
        sgn = 1 if y_peak > y0 else -1

        self.parameters["a"].heuristic = a * sgn
        self.parameters["fwhmh"].heuristic = fwhmh

        cut_off = 2 * tau

        x0 = heuristics.find_x_offset_sym_peak_fft(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=cut_off,
            test_pts=x[peak_idx],
        )

        self.parameters["x0"].heuristic = x0
