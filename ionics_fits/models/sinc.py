from typing import List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..normal import NormalFitter
from ..utils import scale_x, scale_x_inv, scale_y
from . import heuristics
from .rectangle import Rectangle
from .triangle import Triangle


class Sinc(Model):
    """Sinc function according to::

        y = a * sin(w * (x - x0)) / (w * (x - x0)) + y0

    See :meth:`_func` for parameters.
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
        w: ModelParameter(lower_bound=0, scale_func=scale_x_inv()),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param y0: y-axis offset
        :param a: amplitude
        :param w: x scale factor
        """
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * np.sinc(x) + y0
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        self.parameters["y0"].heuristic = np.mean([y[0], y[-1]])
        y0 = self.parameters["y0"].get_initial_value()

        omega, spectrum = heuristics.get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        # Fourier transform of a sinc is a rectangle
        rect = Rectangle()
        rect.parameters["y0"].fixed_to = 0
        rect.parameters["x_l"].fixed_to = 0
        rect.parameters["a"].heuristic = max(abs_spectrum)

        fit = NormalFitter(omega, abs_spectrum, model=rect)

        self.parameters["w"].heuristic = fit.values["x_r"]
        w = self.parameters["w"].get_initial_value() + abs_spectrum[1]

        sgn = 1 if y[np.argmax(np.abs(y - y0))] > y0 else -1
        self.parameters["a"].heuristic = (w / np.pi) * fit.values["a"] * sgn

        x0 = heuristics.find_x_offset_sym_peak_fft(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=w,
        )

        self.parameters["x0"].heuristic = x0


class Sinc2(Model):
    """Sinc-squared function according to::

        y = a * (sin(w * (x - x0)) / (w * (x - x0)))^2 + y0

    See :meth:`_func` for parameters.
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
        w: ModelParameter(lower_bound=0, scale_func=scale_x_inv()),
    ) -> TY:
        """
        :param x0: x-axis offset
        :param y0: y-axis offset
        :param a: amplitude
        :param w: x scale factor
        """
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * (np.sinc(x) ** 2) + y0
        return y

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        y0 = self.parameters["y0"].heuristic = np.mean([y[0], y[-1]])

        omega, spectrum = heuristics.get_spectrum(x, y, trim_dc=True)
        abs_spectrum = np.abs(spectrum)

        # Fourier transform of a sinc^2 is a triangle function
        tri = Triangle()
        tri.parameters["x0"].fixed_to = 0
        tri.parameters["y0"].heuristic = max(abs_spectrum)
        tri.parameters["sym"].fixed_to = 0
        tri.parameters["y_min"].fixed_to = 0

        fit = NormalFitter(omega, abs_spectrum, model=tri)

        intercept = fit.values["y0"] / -fit.values["k"]
        sgn = 1 if y[np.argmax(np.abs(y - y0))] > y0 else -1

        w = self.parameters["w"].get_initial_value(intercept / 2)
        self.parameters["w"].heuristic = w
        self.parameters["a"].heuristic = (
            fit.values["y0"] * sgn * intercept / (2 * np.pi)
        )

        x0 = heuristics.find_x_offset_sym_peak_fft(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=intercept,
        )

        self.parameters["x0"].heuristic = x0
