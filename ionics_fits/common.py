from __future__ import annotations
import dataclasses
import copy
import inspect
import numpy as np
from scipy import signal
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from .utils import Array, ArrayLike


if TYPE_CHECKING:
    num_samples = float
    num_values = float
    num_spectrum_samples = float


@dataclasses.dataclass
class ModelParameter:
    """Metadata associated with a model parameter.

    Attributes:
        lower_bound: lower bound for the parameter. Fitted values are guaranteed to be
            greater than or equal to the lower bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        upper_bound: upper bound for the parameter. Fitted values are guaranteed to be
            lower than or equal to the upper bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        fixed_to: if not `None` the model parameter is fixed to this value during
            fitting instead of being floated.
        initialised_to: if not `None` this value is used as an initial value during
            fitting rather than obtaining a value from the heuristics. This value may
            additionally be used by the heuristics to help find good initial conditions
            for other model parameters where none has been explicitly given.
        scale_func: callable returning a scale factor which the parameter must be
            *multiplied* by if it was fitted using `x` / `y` data that has been
            *multiplied* by the given scale factors. Scale factors are used to improve
            numerical stability by avoiding asking the optimizer to work with very large
            or very small values of `x` and `y`. The callable takes three arguments: the
            x-axis and y-axis scale factors and the model instance. If any `scale_func`
            returns a value that is `0`, not finite (e.g. `nan`) or `None` we do not
            rescale the coordinates. c.f. :meth can_rescale: and :meth rescale:.
    """

    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    fixed_to: Optional[float] = None
    initialised_to: Optional[float] = None
    scale_func: Callable[
        [
            float,
            float,
            Model,
        ],
        Optional[float],
    ] = lambda x_scale, y_scale, model: 1

    def can_rescale(self, x_scale: float, y_scale: float, model: Model) -> bool:
        """Returns `True` if the parameter can be rescaled."""
        scale_factor = self.scale_func(x_scale, y_scale, model)
        return (
            scale_factor is not None and np.isfinite(scale_factor) and scale_factor != 0
        )

    def rescale(self, x_scale: float, y_scale: float, model: Model) -> float:
        """Rescales the parameter metadata based on the specified x and y data scale
        factors and returns the overall scale factor used.
        """
        scale_factor = self.scale_func(x_scale, y_scale, model)

        def _rescale(attr):
            if attr is None:
                return None
            return attr / scale_factor

        self.lower_bound = _rescale(self.lower_bound)
        self.upper_bound = _rescale(self.upper_bound)
        self.fixed_to = _rescale(self.fixed_to)
        self.initialised_to = _rescale(self.initialised_to)

        return scale_factor

    def get_initial_value(self, default: Optional[float] = None) -> Optional[float]:
        """If a value is known for this parameter prior to fitting -- either because an
        initial value has been set or because the parameter has been fixed -- we return
        it, otherwise we return :param default:. The return value is clipped to lie
        between the set lower and upper bounds.

        Does not mutate the parameter.
        """
        if self.fixed_to is not None:
            value = self.fixed_to
        elif self.initialised_to is not None:
            value = self.initialised_to
        else:
            value = default

        if value is not None:
            value = self.clip(value)

        return value

    def clip(self, value: float):
        """Clip value to lie between lower and upper bounds."""
        return np.clip(value, self.lower_bound, self.upper_bound)

    def initialise(self, estimate: float) -> float:
        """Sets the parameter's initial value based on the supplied estimate. If an
        initial value is already known for this parameter (see :meth get_initial_value:)
        we use that instead of the supplied estimate. The value is clipped to lie
        between the set lower and upper bounds.

        After this method, :attribute initialised_to: the parameter either has a valid
        initial value (i.e. one that is not `None` and lies between the set bounds) or
        a `ValueError` be raised.

        :returns: the initialised value
        """
        self.initialised_to = self.get_initial_value(estimate)

        if self.initialised_to is None:
            raise ValueError("No valid initial value set for parameter")

        return self.initialised_to


class Model:
    """Base class for fit models.

    A model groups a function to be fitted with associated metadata (parameter names,
    default bounds etc) and heuristics. It is agnostic about the method of fitting or
    the data statistics.
    """

    def __init__(self, parameters: Optional[Dict[str, ModelParameter]] = None):
        """
        :param parameters: optional dictionary mapping names of model parameters to
            their metadata. This should be `None` (default) if the model has a static
            set of parameters in which case the parameter dictionary is generated from
            the call signature of :meth _func:. The model parameters are stored as
            `self.parameters` and may be modified after construction to change the model
            behaviour during fitting (e.g. to change the bounds, fixed parameters, etc).
        """
        if parameters is None:
            spec = inspect.getfullargspec(self._func)
            for name in spec.args[2:]:
                assert isinstance(
                    spec.annotations[name], ModelParameter
                ), "Model parameters must be instances of `ModelParameter`"
            self.parameters = {name: spec.annotations[name] for name in spec.args[2:]}
        else:
            self.parameters = parameters

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
        return self._func(x, **param_values)

    def _func(
        self,
        x: Array[("num_samples",), np.float64],
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        Overload this in preference to `func` unless the FitModel takes a
        dynamic set of parameters. Use : class ModelParameter: objects as the annotation
        for the parameters arguments. e.g.:

        ```
        def _func(self, x, a: ModelParameter(), b: ModelParameter()):
        ```

        A dictionary of `ModelParameter`s may be accessed via the `self.parameters`
        attribute. These parameters may be modified by the user to change the model
        behaviour during fitting (e.g. to change the bounds, fixed parameters, etc).

        :param x: x-axis data
        :returns: array of model values
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        return {}, {}

    def post_fit(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ):
        """Hook called post-fit. Override to implement custom functionality.

        :param x: x-axis data
        :param y: y-axis data
        :param fitted_params: dictionary mapping model parameter names to their fitted
            values
        :param fit_uncertainties: dictionary mapping model parameter names to
            the fit uncertainties (`0` for fixed parameters).
        """
        pass

    def param_min_sqrs(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        parameters: Dict[str, ModelParameter],
        scanned_param: str,
        scanned_param_values: ArrayLike["num_values", np.float64],
    ) -> Tuple[float, float]:
        """Scans one model parameter while holding the others fixed to find the value
        that gives the best fit to the data (minimum sum-squared residuals).

        :param x: x-axis data
        :param y: y-axis data
        :param parameters: dictionary of model parameters
        :param scanned_param: name of parameter to optimize
        :param scanned_param_values: array of scanned parameter values to test

        :returns: tuple with the value from :param scanned_param_values: which results
        in lowest residuals and the root-sum-squared residuals for that value.
        """
        param_values = {
            param: value
            for param, param_data in parameters.items()
            if (value := param_data.get_initial_value()) is not None
        }

        scanned_param_values = np.asarray(scanned_param_values)
        costs = np.zeros(scanned_param_values.shape)
        for idx, value in np.ndenumerate(scanned_param_values):
            param_values[scanned_param] = value
            y_params = self.func(x, param_values)
            costs[idx] = np.sqrt(np.sum(np.power(y - y_params, 2)))
        opt = np.argmin(costs)
        return float(scanned_param_values[opt]), float(costs[opt])

    def get_spectrum(
        self,
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
        self,
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


class Fitter:
    """Base class for fitters.

    Fitters perform maximum likelihood parameter estimation on a dataset under the
    assumption of a certain model and statistics (normal, binomial, etc) and store the
    results as attributes.

    Attributes:
        x: x-axis data
        y: y-axis data
        sigma: optional y-axis  standard deviations. Only used by `NormalFitter`
        values: dictionary mapping model parameter names to their fitted values
        uncertainties: dictionary mapping model parameter names to their fit
            uncertainties. For sufficiently large datasets, well-formed problems and
            ignoring covariances these are the 1-sigma confidence intervals (roughly:
            there is a 1/3 chance that the real parameter values differ from their
            fitted values by more than this much).
        initial_values: dictionary mapping model parameter names to the initial values
            used to seed the fit.
        model: the fit model
        fit_significance: if applicable, the fit significance as a number between 0 and
            1 (1 indicates perfect agreement between the fitted model and input
            dataset). See :meth _fit_significance: for details.
        free_parameters: list of names of the model parameters floated during the fit
    """

    x: Array[("num_samples",), np.float64]
    y: Array[("num_samples",), np.float64]
    sigma: Optional[Array[("num_samples",), np.float64]]
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model
    fit_significance: Optional[float]
    free_parameters: List[str]

    def __init__(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_samples",), np.float64],
        sigma: Optional[ArrayLike[("num_samples",), np.float64]],
        model: Model,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param sigma: optional y-axis  standard deviations. Only used by `NormalFitter`
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults.
        """
        model = copy.deepcopy(model)

        # sanitize input dataset
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True)
        sigma = None if sigma is None else np.array(sigma, dtype=np.float64, copy=True)

        if x.shape != y.shape:
            raise ValueError("Shapes of x and y must match.")

        if sigma is not None and sigma.shape != y.shape:
            raise ValueError("Shapes of sigma and y must match.")

        valid_pts = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[valid_pts]
        y = y[valid_pts]
        sigma = None if sigma is None else sigma[valid_pts]

        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]
        sigma = None if sigma is None else sigma[inds]

        self.x = x
        self.y = y

        # Rescale coordinates to improve numerics (optimizers need to do things like
        # calculate numerical derivatives which is easiest if x and y are O(1)).
        x_scale = np.max(np.abs(x))
        y_scale = np.max(np.abs(y))

        parameters = copy.deepcopy(model.parameters)
        rescale_coords = all(
            [
                param_data.can_rescale(x_scale, y_scale, model)
                for param_data in parameters.values()
            ]
        )

        if not rescale_coords:
            x_scale = 1
            y_scale = 1
            scale_factors = {param: 1 for param in parameters.keys()}
        else:
            scale_factors = {
                param: param_data.rescale(x_scale, y_scale, model)
                for param, param_data in parameters.items()
            }

        x = x / x_scale
        y = y / y_scale
        sigma = None if sigma is None else sigma / y_scale

        model.estimate_parameters(x, y, parameters)

        # raises an exception if any parameter has not been initialised
        try:
            for param, param_data in parameters.items():
                param_data.initialise(None)
        except ValueError:
            raise Exception(f"No initial value found for parameter {param}.")

        fixed_params = {
            param_name: param_data.fixed_to
            for param_name, param_data in parameters.items()
            if param_data.fixed_to is not None
        }
        self.free_parameters = [
            param_name
            for param_name, param_data in parameters.items()
            if param_data.fixed_to is None
        ]

        if self.free_parameters == []:
            raise ValueError("Attempt to fit with no free parameters")

        def free_func(
            x: Array[("num_samples",), np.float64], *free_param_values: float
        ):
            """Call the model function with the values of the free parameters."""
            params = {
                param: value
                for param, value in zip(self.free_parameters, list(free_param_values))
            }
            params.update(fixed_params)
            y = model.func(x, params)
            return y

        fitted_params, uncertainties = self._fit(x, y, sigma, parameters, free_func)

        fitted_params.update({param: value for param, value in fixed_params.items()})
        uncertainties.update({param: 0 for param in fixed_params.keys()})

        fitted_params = {
            param: value * scale_factors[param]
            for param, value in fitted_params.items()
        }
        uncertainties = {
            param: value * scale_factors[param]
            for param, value in uncertainties.items()
        }
        initial_values = {
            param: param_data.initialised_to * scale_factors[param]
            for param, param_data in parameters.items()
        }

        derived_params, derived_uncertainties = model.calculate_derived_params(
            fitted_params, uncertainties
        )
        fitted_params.update(derived_params)
        uncertainties.update(derived_uncertainties)

        model.post_fit(x, y, fitted_params, uncertainties)

        self.model = model
        self.sigma = sigma
        self.values = fitted_params
        self.uncertainties = uncertainties
        self.initial_values = initial_values
        self.fit_significance = self._fit_significance()

    @staticmethod
    def _fit(
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        sigma: Optional[Array[("num_samples",), np.float64]],
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., Array[("num_samples",), np.float64]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the parameter estimation.

        `Fitter` implementations must override this method to provide a fit with
        appropriate statistics.

        :param x: rescaled x-axis data
        :param y: rescaled y-axis data
        :param sigma: rescaled standard deviations
        :param parameters: dictionary of rescaled model parameters
        :param free_func: convenience wrapper for the model function, taking only values
            for the fit's free parameters

        :returns: tuple of dictionaries mapping model parameter names to their fitted
            values and uncertainties.
        """
        raise NotImplementedError

    def _fit_significance(self) -> Optional[float]:
        """Returns an estimate of the goodness of fit as a number between 0 and 1.

        This is the defined as the probability that fit residuals as large as the ones
        we observe could have arisen through chance given our assumed statistics and
        assuming that the fitted model perfectly represents the probability distribution

        A value of `1` indicates a perfect fit (all data points lie on the fitted curve)
        a value close to 0 indicates significant deviations of the dataset from the
        fitted model.
        """
        return None

    def evaluate(
        self, x_fit: Optional[Union[Array[("num_samples",), np.float64], int]] = None
    ) -> Tuple[
        Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
    ]:
        """Evaluates the model function using the fitted parameter set.

        :param x_fit: optional x-axis points to evaluate the model at. If `None` we use
            the dataset values. If a scalar we generate an axis of linearly spaced
            points between the minimum and maxim value of the x-axis dataset. Otherwise
            it should be an array of x-axis data points to use.

        :returns: tuple of x-axis values used and model values for those points
        """
        x_fit = x_fit if x_fit is not None else self.x

        if np.isscalar(x_fit):
            x_fit = np.linspace(np.min(self.x), np.max(self.x), x_fit)

        y_fit = self.model.func(x_fit, self.values)
        return x_fit, y_fit

    def residuals(self) -> Array[("num_samples",), np.float64]:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]
