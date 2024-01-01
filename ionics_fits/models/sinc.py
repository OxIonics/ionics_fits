from typing import Tuple, TYPE_CHECKING
import numpy as np

from . import heuristics
from .rectangle import Rectangle
from .triangle import Triangle
from .. import common, Model, ModelParameter, NormalFitter
from ..utils import Array

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
        a: ModelParameter(scale_func=common.scale_y),
        w: ModelParameter(lower_bound=0, scale_func=common.scale_x_inv),
    ) -> Array[("num_samples",), np.float64]:
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * np.sinc(x) + y0
        return y

    # pytype: enable=invalid-annotation
    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
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
        w = self.parameters["w"].get_initial_value()

        sgn = 1 if y[np.argmax(np.abs(y - y0))] > y0 else -1
        self.parameters["a"].heuristic = 2 * w * fit.values["a"] * sgn

        x0 = heuristics.find_x_offset_sym_peak(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=w,
        )

        self.parameters["x0"].heuristic = x0


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
        a: ModelParameter(scale_func=common.scale_y),
        w: ModelParameter(lower_bound=0, scale_func=common.scale_x_inv),
    ) -> Array[("num_samples",), np.float64]:
        x = w * (x - x0) / np.pi  # np.sinc(x) = sin(pi*x) / (pi*x)
        y = a * (np.sinc(x) ** 2) + y0
        return y

    # pytype: enable=invalid-annotation

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
    ):
        # Ensure that y is a 1D array
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

        self.parameters["w"].heuristic = 0.5 * intercept
        self.parameters["a"].heuristic = fit.values["y0"] * sgn * intercept

        x0 = heuristics.find_x_offset_sym_peak(
            model=self,
            x=x,
            y=y,
            omega=omega,
            spectrum=spectrum,
            omega_cut_off=intercept,
        )

        self.parameters["x0"].heuristic = x0
