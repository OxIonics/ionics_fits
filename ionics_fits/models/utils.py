import copy
import dataclasses
import numpy as np
from scipy import signal
from typing import Dict, Tuple, TYPE_CHECKING
from ..common import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_spectrum_samples = float


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
        fixed_params: Dict[str, float] = None,
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

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
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
        y: Array[("num_samples",), np.float64],
        inner_parameters: Dict[str, ModelParameter],
    ) -> Dict[str, float]:
        return self.inner.estimate_parameters(x, y, inner_parameters)

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
            model_parameters[new_param].initialise(initial_value)


def get_spectrum(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns the frequency spectrum (Fourier transform) of a dataset.

    :param x: x-axis data
    :param y: y-axis data
    :returns: tuple with the frequency axis (angular units) and Fourier transform of
        the dataset.
    """
    dx = x.ptp() / x.size
    n = x.size
    freq = np.fft.fftfreq(n, dx)
    y_f = np.fft.fft(y, norm="ortho") / np.sqrt(n)

    y_f = y_f[: int(n / 2)]
    freq = freq[: int(n / 2)]
    return freq * (2 * np.pi), y_f


def get_pgram(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns a periodogram for a dataset, converted into amplitude units.

    Based on the Lombe-Scargle periodogram (essentially least-squares fitting of
    sinusoids at different frequencies).

    :param x: x-axis data
    :param y: y-axis data
    :returns: tuple with the frequency axis (angular units) and the periodogram
    """
    min_step = np.min(np.diff(x))
    duration = x.ptp()

    # Nyquist limit does not apply to irregularly spaced data
    # We'll use it as a starting point anyway...
    f_max = 0.5 / min_step
    f_min = 0.25 / duration

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
    pgram = signal.lombscargle(x, y, omega_list, precenter=True)
    pgram = np.sqrt(pgram * 4 / len(y))
    return omega_list, pgram
