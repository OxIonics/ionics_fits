from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from .utils import Array, ArrayLike


if TYPE_CHECKING:
    num_samples = float
    num_values = float
    num_spectrum_pts = float
    num_test_pts = float
    num_y_channels = float


logger = logging.getLogger(__name__)


TX = ArrayLike[("num_samples",), np.float64]
TY = Union[
    Array[("num_y_channels", "num_samples"), np.float64],
    Array[("num_samples"), np.float64],
]


def scale_invariant(x_scale: float, y_scale: float) -> float:
    """Scale function for model parameters whose value is invariant under rescaling of
    the x- and y-axes"""
    return 1


def scale_x(x_scale: float, y_scale: float) -> float:
    """
    Scale function for model parameters whose value scales linearly with the x-axis
    """
    return x_scale


def scale_x_inv(x_scale: float, y_scale: float) -> float:
    """
    Scale function for model parameters whose value scales with 1/x_scale
    """
    return 1 / x_scale


def scale_y(x_scale: float, y_scale: float) -> float:
    """
    Scale function for model parameters whose value scales linearly with the x-axis
    """
    return y_scale


def scale_power(x_power: int, y_power: int) -> Callable[[float, float], float]:
    """Returns a scale function for model parameters whose value scales as a power of
    the x- and y-axes
    """

    def scale_func(x_scale: float, y_scale: float) -> float:
        return (x_scale**x_power) * (y_scale**y_power)

    return scale_func


def scale_undefined(x_scale: float, y_scale: float) -> float:
    """This is typically used when the appropriate scale factor to use must be
    determined at runtime"""
    raise RuntimeError(
        "Attempt to rescale model parameter with undefined scale function"
    )


@dataclasses.dataclass
class ModelParameter:
    """Metadata associated with a model parameter.

    Attributes:
        scale_func: callable returning a scale factor which the parameter must be
            *multiplied* by if it was fitted using `x` / `y` data that has been
            *multiplied* by the given scale factors. Scale factors are used to improve
            numerical stability by avoiding asking the optimizer to work with very large
            or very small values of `x` and `y`. The callable takes the x-axis and
            y-axis scale factors as arguments. A number of default scale functions are
            provided for convenience.
        lower_bound: lower bound for the parameter. Fitted values are guaranteed to be
            greater than or equal to the lower bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        upper_bound: upper bound for the parameter. Fitted values are guaranteed to be
            lower than or equal to the upper bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        fixed_to: if not `None`, the model parameter is fixed to this value during
            fitting instead of being floated. This value may additionally be used by
            the heuristics to help find good initial values for other model parameters
            for which none have been provided by the user. The value of `fixed_to` must
            lie within the bounds of the parameter.
        user_estimate: if not `None` and the parameter is not fixed, this value is
            used as an initial value during fitting rather than obtaining a value from
            the heuristics. This value may additionally be used by the heuristics to
            help find good initial values for other model parameters for which none
            have been provided by the user. The value of `user_estimate` must lie
            within the bounds of the parameter.
        heuristic: if both of `fixed_to` and `user_estimate` are `None`, this value is
            used as an initial value during fitting. It is set by the
            `estimate_parameters` method of the model in which the parameter is used
            and should not be set by the user.
    """

    scale_func: Callable[[float, float], float]
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    fixed_to: Optional[float] = None
    user_estimate: Optional[float] = None
    heuristic: Optional[float] = None
    scale_factor: Optional[float] = dataclasses.field(init=False, default=None)

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name == "scale_factor":
            return attr

        scale_factor = self.scale_factor
        if attr is None or scale_factor is None:
            return attr

        if name in [
            "lower_bound",
            "upper_bound",
            "fixed_to",
            "user_estimate",
            "heuristic",
        ]:
            attr /= scale_factor

        return attr

    def __setattr__(self, name, value):
        scale_factor = self.scale_factor

        if None not in [scale_factor, value] and name in [
            "lower_bound",
            "upper_bound",
            "fixed_to",
            "user_estimate",
            "heuristic",
        ]:
            value *= scale_factor
        object.__setattr__(self, name, value)

    def rescale(self, x_scale: float, y_scale: float):
        """Rescales the parameter metadata based on the specified x and y data scale
        factors.
        """
        if self.scale_factor is not None:
            raise RuntimeError("Attempt to rescale an already rescaled model parameter")
        self.scale_factor = self.scale_func(x_scale, y_scale)

    def unscale(self):
        """Disables rescaling of the parameter metadata"""
        if self.scale_factor is None:
            raise RuntimeError(
                "Attempt to unscale a model parameter which was not rescaled."
            )
        self.scale_factor = None

    def get_initial_value(self, default: Optional[float] = None) -> float:
        """
        Get initial value.

        For fixed parameters, this is the value the parameter is fixed to. For floated
        parameters, it is the value used to seed the fit. In the latter case, the
        initial value is retrieved from `user_estimate` if that attribute is not
        `None`, otherwise `heuristic` is used.

        :param default: optional value to use if no other value is available
        """
        if self.fixed_to is not None:
            value = self.fixed_to
            if self.user_estimate is not None:
                raise ValueError(
                    "User estimates must not be provided for fixed parameters"
                )
        elif self.user_estimate is not None:
            value = self.user_estimate
        elif self.heuristic is not None:
            value = self.clip(self.heuristic)
        elif default is not None:
            value = self.clip(default)
        else:
            raise ValueError("No initial value specified")

        if value < self.lower_bound or value > self.upper_bound:
            raise ValueError("Initial value outside bounds.")

        return value

    def has_initial_value(self) -> bool:
        """
        Returns True if the parameter is fixed, has a user estimate or a heuristic.
        """
        values = [self.fixed_to, self.user_estimate, self.heuristic]
        return any([None is not value for value in values])

    def has_user_initial_value(self) -> bool:
        """Returns True if the parameter is fixed or has a user estimate"""
        return self.fixed_to is not None or self.user_estimate is not None

    def clip(self, value: float) -> float:
        """Clip value to lie between lower and upper bounds."""
        return np.clip(value, self.lower_bound, self.upper_bound)


class Model:
    """Base class for fit models.

    A model groups a function to be fitted with associated metadata (parameter names,
    default bounds etc) and heuristics. It is agnostic about the method of fitting or
    the data statistics.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, ModelParameter]] = None,
        internal_parameters: Optional[List[ModelParameter]] = None,
    ):
        """
        :param parameters: optional dictionary mapping names of model parameters to
            their metadata. This should be `None` (default) if the model has a static
            set of parameters in which case the parameter dictionary is generated from
            the call signature of :meth _func:. The model parameters are stored as
            `self.parameters` and may be modified after construction to change the model
            behaviour during fitting (e.g. to change the bounds, fixed parameters, etc).
        :param internal_parameters: optional list of "internal" model parameters, which
            are not exposed to the user as arguments of :meth func:. Internal parameters
            are rescaled in the same way as regular model parameters, but are not
            otherwise used by :class Model:. These are typically used by models which
            encapsulate / modify the behaviour of other models.
        """
        if parameters is None:
            spec = inspect.getfullargspec(self._func)
            for name in spec.args[2:]:
                assert isinstance(
                    spec.annotations[name], ModelParameter
                ), "Model parameters must be instances of `ModelParameter`"
            self.parameters = {
                name: copy.deepcopy(spec.annotations[name]) for name in spec.args[2:]
            }
        else:
            self.parameters = parameters
        self.internal_parameters = internal_parameters or []

    def __call__(self, x: TX, **kwargs: float) -> TY:
        """Evaluates the model.

        - keyword arguments specify values for model parameters (see model definition)
        - all model parameters which are not `fixed_to` a value by default must be
          specified
        - any parameters which are not specified default to their `fixed_to` values
        """
        args = {
            param_name: param_data.fixed_to
            for param_name, param_data in self.parameters.items()
            if param_data.fixed_to is not None
        }
        args.update(kwargs)
        return self.func(x, args)

    def can_rescale(self) -> Tuple[bool, bool]:
        """
        Returns a Tuple of bools specifying whether the model can be rescaled along
        the x- and y-axes
        """
        raise NotImplementedError

    def rescale(self, x_scale: float, y_scale: float):
        """Rescales the model parameters based on the specified x and y data scale
        factors.

        :param x_scale: x-axis scale factor
        :param y_scale: y-axis scale factor
        """
        for param_name, param_data in self.parameters.items():
            if param_data.scale_func == scale_undefined:
                raise RuntimeError(
                    f"Parameter {param_name} has an undefined scale function"
                )
            param_data.rescale(x_scale, y_scale)

        for param_data in self.internal_parameters:
            param_data.rescale(x_scale, y_scale)

    def unscale(self):
        """Disables rescaling of the model parameters."""
        parameters = list(self.parameters.values()) + self.internal_parameters
        for param_data in parameters:
            param_data.unscale()

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model."""
        raise NotImplementedError

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        To use the model as a function outside of a fit, :meth __call__: generally
        provides a more convenient interface.

        Overload this to provide a model function with a dynamic set of parameters,
        otherwise prefer to override `_func`.

        :param x: x-axis data
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        return self._func(x, **param_values)

    def _func(
        self,
        x: TX,
    ) -> TY:
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
        x: TX,
        y: TY,
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If the model allows rescaling then rescaled
        units will be used everywhere (`x` and `y` as well as parameter values).

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        """
        raise NotImplementedError

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being directly part of the fit) based on values of the fitted parameters and
        their uncertainties.

        :param x: x-axis data
        :param y: y-axis data
        :param fitted_params: dictionary mapping model parameter names to their
            fitted values
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        return {}, {}


class Fitter:
    """Base class for fitters.

    Fitters perform maximum likelihood parameter estimation on a dataset under the
    assumption of a certain model and statistics (normal, binomial, etc) and store the
    results as attributes.

    Attributes:
        x: 1D ndarray of shape (num_samples,) containing x-axis values of valid points.
        y: 2D ndarray of shape (num_y_channels, num_samples) containing y-axis values
            of valid points.
        sigma: 2D ndarray of shape (num_y_channels, num_samples) containing y-axis
            standard errors where available.
        values: dictionary mapping model parameter names to their fitted values
        uncertainties: dictionary mapping model parameter names to their fit
            uncertainties. For sufficiently large datasets, well-formed problems and
            ignoring covariances these are the 1-sigma confidence intervals (roughly:
            there is a 1/3 chance that the real parameter values differ from their
            fitted values by more than this much)
        derived_values: dictionary mapping names of derived parameters (parameters which
            are not part of the fit, but are calculated by the model from the fitted
            parameter values) to their values
        derived_uncertainties: dictionary mapping names of derived parameters to their
            fit uncertainties
        initial_values: dictionary mapping model parameter names to the initial values
            used to seed the fit.
        model: the fit model
        free_parameters: list of names of the model parameters floated during the fit
        x_scale: the applied x-axis scale factor
        y_scale: the applied y-axis scale factor
    """

    x: TX
    y: TY
    sigma: TY
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    derived_values: Dict[str, float]
    derived_uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model
    free_parameters: List[str]
    x_scale: float
    y_scale: float

    def __init__(
        self,
        x: TX,
        y: TY,
        model: Model,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults. The
            model is (deep) copied and stored as an attribute.
        """
        self.model = copy.deepcopy(model)

        self.x = np.array(x, dtype=np.float64, copy=True)
        self.y = np.atleast_2d(np.array(y, dtype=np.float64, copy=True))

        self.sigma = self.calc_sigma()
        self.sigma = np.atleast_2d(self.sigma) if self.sigma is not None else None

        if self.x.ndim != 1:
            raise ValueError("x-axis data must be a 1D array.")

        if self.y.ndim != 2:
            raise ValueError("y-axis data must be a 1D or 2D array.")

        if self.x.shape[0] != self.y.shape[1]:
            raise ValueError(
                "Number of x-axis and y-axis samples must match "
                f"(got {self.x.shape} and {self.y.shape})."
            )

        if self.y.shape[0] != self.model.get_num_y_channels():
            raise ValueError(
                f"Expected {self.model.get_num_y_channels()} y channels, "
                f"got {self.y.shape[0]}."
            )

        if self.sigma is not None and self.sigma.shape != self.y.shape:
            raise ValueError(
                f"Shapes of sigma and y must match (got {self.sigma.shape} and "
                f"{self.y.shape})."
            )

        valid_x = np.isfinite(self.x)
        valid_y = np.all(np.isfinite(self.y), axis=0)
        valid_pts = np.logical_and(valid_x, valid_y).squeeze()

        (inds,) = np.where(valid_pts)
        sorted_inds = inds[np.argsort(self.x[valid_pts])]

        self.x = self.x[sorted_inds]
        self.y = self.y[:, sorted_inds]

        if self.sigma is not None:

            self.sigma = self.sigma[:, sorted_inds]
            if np.any(self.sigma == 0) or not np.all(np.isfinite(self.sigma)):
                raise RuntimeError(
                    "Dataset contains points with zero or infinite uncertainty."
                )

        # Rescale coordinates to improve numerics (optimizers need to do things like
        # calculate numerical derivatives which is easiest if x and y are O(1)).
        #
        # Currently we use a single scale factor for all y channels. This may change in
        # future
        rescale_x, rescale_y = self.model.can_rescale()
        self.x_scale = np.max(np.abs(self.x)) if rescale_x else 1.0
        self.y_scale = np.max(np.abs(self.y)) if rescale_y else 1.0

        # Corner-case if a y-channel has values that are all 0
        if self.y_scale == 0 or not np.isfinite(self.y_scale):
            self.y_scale = 1
            rescale_y = False

        self.model.rescale(self.x_scale, self.y_scale)

        x_scaled = self.x / self.x_scale
        y_scaled = self.y / self.y_scale

        self.model.estimate_parameters(x_scaled, y_scaled)

        for param, param_data in self.model.parameters.items():
            if not param_data.has_initial_value():
                raise RuntimeError(
                    "No fixed_to, user_estimate or heuristic specified"
                    f" for parameter `{param}`."
                )

        self.fixed_parameters = {
            param_name: param_data.fixed_to
            for param_name, param_data in self.model.parameters.items()
            if param_data.fixed_to is not None
        }
        self.free_parameters = [
            param_name
            for param_name, param_data in self.model.parameters.items()
            if param_data.fixed_to is None
        ]

        if self.free_parameters == []:
            raise ValueError("Attempt to fit with no free parameters.")

        def free_func(
            x: Array[("num_samples",), np.float64], *free_param_values: float
        ) -> Array[("num_y_channels", "num_samples"), np.float64]:
            """Call the model function with the values of the free parameters."""
            params = {
                param: value
                for param, value in zip(self.free_parameters, list(free_param_values))
            }
            params.update(self.fixed_parameters)
            y = self.model.func(x, params)
            return y

        fitted_params, uncertainties = self._fit(
            x_scaled, y_scaled, self.model.parameters, free_func
        )
        fitted_params.update(
            {param: value for param, value in self.fixed_parameters.items()}
        )
        uncertainties.update({param: 0 for param in self.fixed_parameters.keys()})

        self.values = {
            param: value * self.model.parameters[param].scale_factor
            for param, value in fitted_params.items()
        }
        self.uncertainties = {
            param: value * self.model.parameters[param].scale_factor
            for param, value in uncertainties.items()
        }

        self.model.unscale()

        derived = self.model.calculate_derived_params(
            self.x, self.y, self.values, self.uncertainties
        )
        self.derived_values, self.derived_uncertainties = derived

        self.initial_values = {
            param: param_data.get_initial_value()
            for param, param_data in self.model.parameters.items()
        }

    def _fit(
        self,
        x: TX,
        y: TY,
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., TY],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the parameter estimation.

        `Fitter` implementations must override this method to provide a fit with
        appropriate statistics.

        :param x: rescaled x-axis data, must be a 1D array
        :param y: rescaled y-axis data
        :param parameters: dictionary of rescaled model parameters
        :param free_func: convenience wrapper for the model function, taking only values
            for the fit's free parameters

        :returns: tuple of dictionaries mapping model parameter names to their fitted
            values and uncertainties.
        """
        raise NotImplementedError

    def evaluate(
        self,
        transpose_and_squeeze=False,
        x_fit: Optional[TX] = None,
    ) -> Tuple[TX, TY]:
        """Evaluates the model function using the fitted parameter set.

        :param transpose_and_squeeze: if True, array `y_fit` is transposed
            and squeezed before being returned. This is intended to be used
            for plotting, since matplotlib requires different y-series to be
            stored as columns.
        :param x_fit: optional x-axis points to evaluate the model at. If
            `None`, we use the values stored as attribute `x` of the fitter.

        :returns: tuple of x-axis values used and corresponding y-axis values
            of the fitted model
        """
        x_fit = x_fit if x_fit is not None else self.x
        y_fit = self.model.func(x_fit, self.values)

        if transpose_and_squeeze:
            return x_fit, y_fit.T.squeeze()
        return x_fit, y_fit

    def residuals(self) -> TY:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]

    def calc_sigma(
        self,
    ) -> TY:
        """Return an array of standard error values for each y-axis data point."""
        raise NotImplementedError
