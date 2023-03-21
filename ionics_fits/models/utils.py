import copy
import dataclasses
import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING, TypeVar
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


class MappedModel(Model):
    """`Model` wrapping another `Model` with renamed parameters"""

    def __init__(
        self,
        inner: Model,
        mapped_params: Dict[str, str],
        fixed_params: Optional[Dict[str, float]] = None,
    ):
        """Init

        :param inner: The wrapped model, the implementation of `inner` will be used
            after the parameter mapping has been done.
        :param mapped_params: dictionary mapping names of parameters in the new
            model to names of parameters used in the wrapped model.
        :param fixed_params: dictionary mapping names of parameters used in the
            wrapped model to values they are fixed to in the new model. These
            will not be parameters of the new model.
        """
        inner_params = inner.parameters

        if fixed_params is None:
            fixed_params = {}

        if unknown_mapped_params := set(mapped_params.values()) - inner_params.keys():
            raise ValueError(
                "The target of parameter mappings must be parameters of the inner "
                f"model. The mapping targets are not: {unknown_mapped_params}"
            )

        if unknown_fixed_params := fixed_params.keys() - inner_params.keys():
            raise ValueError(
                "Fixed parameters must be parameters of the inner model. The "
                f"follow fixed parameters are not: {unknown_fixed_params}"
            )

        if missing_params := inner_params.keys() - (
            fixed_params.keys() | mapped_params.values()
        ):
            raise ValueError(
                "All parameters of the inner model must be either mapped of "
                "fixed. The following inner model parameters are neither: "
                f"{missing_params}"
            )

        if duplicated_params := fixed_params.keys() & mapped_params.values():
            raise ValueError(
                "Parameters cannot be both mapped and fixed. The following "
                f"parameters are both: {duplicated_params}"
            )

        params = {
            new_name: inner_params[old_name]
            for new_name, old_name in mapped_params.items()
        }
        super().__init__(parameters=params)
        self.inner = inner
        self.mapped_args = mapped_params
        self.fixed_params = fixed_params or {}

    def can_rescale(self, x_scale: float, y_scale: float) -> bool:
        """Returns True if the model can be rescaled"""
        return self.inner.can_rescale(x_scale, y_scale)

    @staticmethod
    def get_scaled_model(model, x_scale: float, y_scale: float):
        """Returns a scaled copy of a given model object

        :param model: model to be copied and rescaled
        :param x_scale: x-axis scale factor
        :param y_scale: y-axis scale factor
        :returns: a scaled copy of model
        """
        scaled_model = copy.deepcopy(model)
        for param_name, param in scaled_model.inner.parameters.items():
            param.rescale(x_scale, y_scale, scaled_model.inner)

        for fixed_param in scaled_model.fixed_params.keys():
            scale_factor = scaled_model.inner.parameters[fixed_param].scale_factor
            scaled_model.fixed_params[fixed_param] /= scale_factor

        # Expose the scale factors to the fitter so it knows how to rescale the results
        for new_name, old_name in scaled_model.mapped_args.items():
            scale_factor = scaled_model.inner.parameters[old_name].scale_factor
            scaled_model.parameters[new_name].scale_factor = scale_factor

        return scaled_model

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples", "num_y_channels"), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        Overload this to provide a model function with a dynamic set of parameters,
        otherwise prefer to override `_func`.

        :param x: x-axis data
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        new_params = {
            old_name: param_values[new_name]
            for new_name, old_name in self.mapped_args.items()
        }
        new_params.update(self.fixed_params)
        return self.inner.func(x, new_params)

    def _inner_estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        inner_parameters: Dict[str, ModelParameter],
    ) -> Dict[str, float]:
        return self.inner.estimate_parameters(x, y, inner_parameters)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        inner_parameters = {
            original_param: copy.deepcopy(model_parameters[new_param])
            for new_param, original_param in self.mapped_args.items()
        }

        inner_parameters.update(
            {
                param: ModelParameter(
                    lower_bound=value, upper_bound=value, fixed_to=value
                )
                for param, value in self.fixed_params.items()
            }
        )

        self._inner_estimate_parameters(x, y, inner_parameters)

        for new_param, original_param in self.mapped_args.items():
            initial_value = inner_parameters[original_param].get_initial_value()
            model_parameters[new_param].heuristic = initial_value


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

    :param x: x-axis data
    :param y: y-axis data. For models with multiple y channels, this should contain
        data from a single channel only.
    :param density_units: if `False` we apply normalization for narrow-band signals. If
        `True` we normalize for continuous distributions.
    :param trim_dc: if `True` we do not return the DC component.
    """
    if y.ndim != 1:
        raise ValueError(
            f"{y.shape[1]} y channels were provided to a method which takes 1"
        )

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
    if y.ndim != 1:
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
