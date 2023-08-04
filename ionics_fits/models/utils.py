import dataclasses
import numpy as np
from scipy import signal
from typing import Dict, Tuple, Type, TYPE_CHECKING, TypeVar
from ..common import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_spectrum_samples = float
    num_y_channels = float

TModel = TypeVar("TModel", bound=Type[Model])


@dataclasses.dataclass
class PeriodicModelParameter(ModelParameter):
    period: float = 1
    offset: float = 0
    lower_bound: float = dataclasses.field(init=False)
    upper_bound: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.lower_bound = 1.5 * self.offset
        self.upper_bound = 1.5 * (self.offset + self.period)

    upper_bound = 1.5 * np.pi

    def clip(self, value: float):
        """Clip value to lie between lower and upper bounds."""
        value = value - self.offset
        return (value % self.period) + self.offset


def rescale_model_x(model_class: TModel, x_scale: float) -> TModel:
    """Rescales the x-axis for a model class.

    This is commonly used to convert models between linear and angular units.

    :param model_class: model class to rescale
    :param x_scale: multiplicative x-axis scale factor. To convert a model that takes
      x in angular units and convert to one that takes x in linear units use
      `x_scale = 2 * np.pi`
    """

    class ScaledModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__x_scale = x_scale
            self.__rescale = True

        def func(
            self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
        ) -> Array[("num_samples", "num_y_channels"), np.float64]:
            x = (x * self.__x_scale) if self.__rescale else x
            return super().func(x, param_values)

        def estimate_parameters(
            self,
            x: Array[("num_samples",), np.float64],
            y: Array[("num_samples", "num_y_channels"), np.float64],
            model_parameters: Dict[str, ModelParameter],
        ):
            # avoid double rescaling if estimate_parameters calls self.func internally
            self.__rescale = False
            super().estimate_parameters(x * self.__x_scale, y, model_parameters)
            self.__rescale = True

    ScaledModel.__name__ = model_class.__name__
    ScaledModel.__doc__ = model_class.__doc__

    return ScaledModel


def get_spectrum(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
    density_units: bool = True,
    trim_dc: bool = False,
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns the frequency spectrum (Fourier transform) of a dataset.

    :param x: 1D ndarray of shape (num_samples,) containing x-axis data
    :param y: 1D ndarray of shape (num_samples,) containing y-axis data
    :param density_units: if `False` we apply normalization for narrow-band signals. If
        `True` we normalize for continuous distributions.
    :param trim_dc: if `True` we do not return the DC component.
    """
    if x.ndim != 1:
        raise ValueError("x-axis data must be a 1D array.")
    if y.ndim != 1:
        raise ValueError("y-axis data must be a 1D array.")

    dx = x.ptp() / x.size
    n = x.size
    omega = np.fft.fftfreq(n, dx) * (2 * np.pi)
    y_f = np.fft.fft(y, norm="ortho") / np.sqrt(n)

    y_f = y_f[: int(n / 2)]
    omega = omega[: int(n / 2)]

    if density_units:
        d_omega = 2 * np.pi / (n * dx)
        y_f /= d_omega

    if trim_dc:
        omega = omega[1:]
        y_f = y_f[1:]

    return omega, y_f


def get_pgram(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
    density_units: bool = True,
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns a periodogram for a dataset, converted into amplitude units.

    Based on the Lombe-Scargle periodogram (essentially least-squares fitting of
    sinusoids at different frequencies).

    :param x: x-axis data
    :param y: y-axis data. For models with multiple y channels, this should contain
        data from a single channel only.
    :param density_units: if `False` (default) we apply normalization for narrow-band
        signals. If `True` we normalize for continuous distributions.
    :returns: tuple with the frequency axis (angular units) and the periodogram
    """
    if y.ndim != 1 and y.shape[1] > 1:
        raise ValueError(
            f"{y.shape[1]} y channels were provided to a method which takes 1"
        )

    dx = np.min(np.diff(x))
    duration = x.ptp()
    n = int(duration / dx)
    d_omega = 2 * np.pi / (n * dx)

    # Nyquist limit does not apply to irregularly spaced data
    # We'll use it as a starting point anyway...
    f_nyquist = 0.5 / dx

    omega_list = 2 * np.pi * np.linspace(d_omega, f_nyquist, n)
    pgram = signal.lombscargle(x, y, omega_list, precenter=True)
    pgram = np.sqrt(np.abs(pgram) * 4 / len(y))

    if density_units:
        pgram /= d_omega

    return omega_list, pgram
