import dataclasses
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from .utils import Array, ArrayLike

if TYPE_CHECKING:
    num_samples = float
    num_values = float


@dataclasses.dataclass
class FitParameter:
    """Describes a single fit parameter.

    Arguments:
        bounds: tuple of default `(lower, upper)` bounds for the parameter. Fitted
            values are guaranteed to lie between lower and upper.
        fixed_to: optional float specifying the value (if any) that the parameter is
            fixed to by default. All parameters which are not explicitly fixed are
            floated during the fit.
        scale_func: callable returning a scale factor which the parameter must be
            *multiplied* by if it was fitted using `x` / `y` data *divided* by the given
            scale factors. This is used to improve numerics by avoiding asking
            the optimizer to work with very large or very small values of `x` and `y`.
            The callable takes three arguments: the x-axis scale factor, the y-axis
            scale factor and a dictionary of fixed parameter values. If any `scale_func`
            returns a value that is `0`, not finite (e.g. `nan`) or `None` we do not
            rescale the coordinates.
    """

    bounds: Tuple[float, float] = (-np.inf, np.inf)
    fixed_to: Optional[float] = None
    scale_func: Callable[
        [
            Array[("num_samples",), np.float64],
            Array[("num_samples",), np.float64],
            Dict[str, float],
        ],
        Optional[float],
    ] = lambda x_scale, y_scale, fixed_params: 1


class FitModel:
    """Base class for fit models.

    A model represents a function to be fitted and associated metadata (parameter names
    and default bounds) and heuristics.
    """

    _PARAMETERS: Dict[str, FitParameter] = {}

    @classmethod
    def param_min_sqrs(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        params: Dict[str, float],
        scanned_param: str,
        scanned_param_values: ArrayLike["num_values", np.float64],
    ) -> Tuple[float, float]:
        """Tests an array of values for one model parameter to find the value which
        results in lowest sum-sqaured residuals given fixed values for all other model
        parameters.

        :param x: x-axis data
        :param y: y-axis data
        :param params: dictionary of fixed parameter values
        :param scanned_param: name of parameter to optimize
        :param scanned_param_values: array of scanned parameter values to test

        :returns: tuple with the value from :param scanned_param_values: which results
        in lowest residuals an the root-sum-squared residuals for that value.
        """
        params = dict(params)
        scanned_param_values = np.asarray(scanned_param_values)
        costs = np.zeros(scanned_param_values.shape)
        for idx, value in np.ndenumerate(scanned_param_values):
            params[scanned_param] = value
            y_params = cls.func(x, params)
            costs[idx] = np.sqrt(np.sum(np.power(y - y_params, 2)))
        opt = np.argmin(costs)
        return float(scanned_param_values[opt]), float(costs[opt])

    @classmethod
    def pre_fit(
        cls,
        fixed_params: Dict[str, float],
        initial_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        free_params: List[str],
    ):
        """Hook called post-fit, override to implement custom functionality.

        :param fixed_params: dictionary mapping names of fixed (not floated) parameters
            to their values.
        :param initial_values: dictionary mapping names of parameters with
            user-specified initial values to those values.
        :param bounds: dictionary mapping model parameter names to a tuple of
            `(lower, upper)` parameter bounds. Fitted parameter values are guaranteed
            to lie between lower and upper bounds.
        :param free_params: list of names of the model's free (not fixed) parameters.
        """
        pass

    @classmethod
    def post_fit(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        p_fit: Dict[str, float],
        p_err: Dict[str, float],
    ):
        """Hook called post-fit, override to implement custom functionality.

        :param x: x-axis data
        :param y: y-axis data
        :param p_fit: dictionary mapping model parameter names to their fitted values
        :param p_err: dictionary mapping model parameter names to the fit uncertainties
            (`0` for fixed parameters).
        """
        pass

    @classmethod
    def func(
        cls, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given
        parameter set and returns the result.

        :param x: x-axis data
        :param params: dictionary of parameter values
        :returns: array of model values
        """
        raise NotImplementedError

    @staticmethod
    def calculate_derived_params(
        fitted_params: Dict[str, float], uncertainties: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns dictionaries of values and uncertainties for the derived model
        parameters (parameters which are calculated from the fit results rather than
        being part of the fit themselves) based on values of the fitted parameters and
        their uncertainties.

        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param uncertainties: dictionary mapping model parameter names to their fit
            uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        return {}, {}

    @classmethod
    def get_parameters(cls) -> Dict[str, FitParameter]:
        """Returns a dictionary mapping model parameter names to their metadata."""
        return dict(cls._PARAMETERS)

    @classmethod
    def estimate_parameters(
        cls,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the model parameter values for the
        specified dataset.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values, typically called as part of `FitBase.fit`.

        :param x: x-axis data
        :param y: y-axis data
        :param known_values: dictionary mapping model parameter names to values
            parameters whose value is known (e.g. because the parameter is fixed to a
            certain value or an initial value has been provided by the user).
        :param bounds: dictionary of parameter bounds. Estimated values will be clipped
            to lie within bounds.
        """
        raise NotImplementedError


class FitBase:
    def __init__(
        self,
        model: FitModel,
        param_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
        fixed_params: Optional[Dict[str, Optional[float]]] = None,
        initial_values: Optional[Dict[str, float]] = None,
    ):
        """
        :param model: the model function to fit to.
        :param param_bounds: dictionary mapping model parameter names to tuples of
            `(lower, upper)` bounds. Entries in this dictionary override the defaults
            provided by the model. To unbound a parameter which is bounded by default
            in the model, pass either `None` or `(-np.inf, np.inf)` as bounds.
        :param fixed_params: dictionary mapping model parameter names to values which
            they are to be fixed to. Fixed parameters are not floated during the fit.
            Entries in this dictionary override the defaults provided by the model. By
            default, models float all commonly used parameters and only fix rarely used
            ones. To float a parameter that is fixed by default in the model, provide
            `None` as a value.
        :param initial_values: dictionary mapping parameter names to initial values.
            These values are used by the heuristics. If you find you need these values
            it may be indicative of an issue, such as: the mode heuristic needing more
            work (in which case please consider filing an issue); a poor-quality
            dataset; too many floated parameters; etc.
        """
        self._model = model
        params = model.get_parameters()
        self._param_names = set(params.keys())

        default_bounds = {
            param: param_data.bounds for param, param_data in params.items()
        }
        default_fixed = {
            param: param_data.fixed_to
            for param, param_data in params.items()
            if param_data.fixed_to is not None
        }

        def validate_param_names(params, kind):
            param_names = set(params.keys())
            invalid = param_names - self._param_names
            if invalid:
                raise ValueError(f"Invalid {kind} parameter names: {invalid}")

        param_bounds = param_bounds or {}
        fixed_params = fixed_params or {}
        initial_values = initial_values or {}

        validate_param_names(param_bounds, "parameter bound")
        validate_param_names(fixed_params, "fixed parameter")
        validate_param_names(initial_values, "initial value")

        self._param_bounds = default_bounds
        self._fixed_params = default_fixed
        self._initial_values = {}

        self._param_bounds.update(param_bounds)
        self._fixed_params.update(fixed_params)
        self._initial_values.update(initial_values)

        self._param_bounds = {
            param: np.asarray(bounds if bounds is not None else (-np.inf, np.inf))
            for param, bounds in self._param_bounds.items()
        }
        self._fixed_params = {
            param: fixed
            for param, fixed in self._fixed_params.items()
            if fixed is not None
        }
        self._initial_values = {
            param: initial
            for param, initial in self._initial_values
            if initial is not None
        }

        self._free_params = self._param_names - set(self._fixed_params.keys())

        self._estimated_values = None
        self._fitted_params = None
        self._fitted_param_uncertainties = None

    def set_dataset(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_samples",), np.float64],
    ):
        """Sets the dataset to be fit.

        :param x: x-axis data
        :param y: y-axis data
        """
        self._x = np.array(x, dtype=np.float64, copy=True)
        self._y = np.array(y, dtype=np.float64, copy=True)

        valid_pts = np.logical_and(np.isfinite(self._x), np.isfinite(self._y))
        self._x = self._x[valid_pts]
        self._y = self._y[valid_pts]

        if self._x.shape != self._y.shape:
            raise ValueError("Shapes of x and y must match.")

        inds = np.argsort(self._x)
        self._x = self._x[inds]
        self._y = self._y[inds]

        self._estimated_values = None
        self._fitted_params = None
        self._fitted_param_uncertainties = None

    def fit(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Fit the dataset and return dictionaries mapping model parameter names, including
        derived parameters, to their fitted parameter values and uncertainties. Fixed
        parameters are given an uncertainty of `0`.
        """
        if self._x is None:
            raise ValueError("Cannot fit without first setting a dataset")

        if not self._free_params:
            raise ValueError("Attempt to fit without any floated parameters")

        # We want to be able to mutate these without altering the instance attributes
        x = np.array(self._x, copy=True)
        y = np.array(self._y, copy=True)

        bounds = dict(self._param_bounds)
        fixed_params = dict(self._fixed_params)
        initial_values = dict(self._initial_values)
        initial_values.update(self._fixed_params)
        free_params = list(self._free_params)

        # Rescale our coordinates to avoid working with very large/small values
        x_scale = np.max(np.abs(self._x))
        y_scale = np.max(np.abs(self._y))

        scale_funcs = {
            param: param_data.scale_func
            for param, param_data in self._model.get_parameters().items()
        }

        assert set(scale_funcs.keys()) == set(self._param_names)

        scale_factors = {
            param: scale_func(x_scale, y_scale, fixed_params)
            for param, scale_func in scale_funcs.items()
        }
        rescale_coords = all(
            [
                scale is not None and np.isfinite(scale) and scale != 0
                for scale in scale_factors.values()
            ]
        )

        if not rescale_coords:
            x_scale = None
            y_scale = None
        else:
            x /= x_scale
            y /= y_scale

            initial_values = {
                param: value / scale_factors[param]
                for param, value in initial_values.items()
            }
            fixed_params = {
                param: value / scale_factors[param]
                for param, value in fixed_params.items()
            }
            bounds = {
                param: value / scale_factors[param] for param, value in bounds.items()
            }

        self._model.pre_fit(
            fixed_params=fixed_params,
            initial_values=initial_values,
            bounds=bounds,
            free_params=free_params,
        )

        # Make sure we're using the known values and clip to bounds
        estimated_values = self._model.estimate_parameters(x, y, initial_values, bounds)
        estimated_values.update(initial_values)
        for param, value in estimated_values.items():
            lower, upper = bounds[param]
            estimated_values[param] = min(max(value, lower), upper)

        initial_values = estimated_values
        self._estimated_values = estimated_values  # stored for debugging purposes

        # Trim out free factors in preparation for the optimization step
        bounds = {
            param: bounds for param, bounds in bounds.items() if param in free_params
        }
        initial_values = {
            param: value
            for param, value in initial_values.items()
            if param in free_params
        }

        def free_func(
            x: Array[("num_samples",), np.float64], *free_param_values: float
        ):
            """Call the model function with the values of the free parameters."""
            params = {
                param: value
                for param, value in zip(free_params, list(free_param_values))
            }
            params.update(fixed_params)
            return self._model.func(x, params)

        p_fit, p_err = self._fit(
            x, y, initial_values, bounds, free_func, x_scale, y_scale
        )

        if rescale_coords:
            p_fit = {
                param: value * scale_factors[param] for param, value in p_fit.items()
            }
            p_err = {
                param: value * scale_factors[param] for param, value in p_err.items()
            }
            self._estimated_values = {
                param: value * scale_factors[param]
                for param, value in self._estimated_values.items()
            }

        p_fit.update({param: value for param, value in self._fixed_params.items()})
        p_err.update({param: 0 for param, _ in self._fixed_params.items()})

        self._model.post_fit(x, y, p_fit, p_err)

        p_derived, p_derived_err = self._model.calculate_derived_params(p_fit, p_err)
        p_fit.update(p_derived)
        p_err.update(p_derived_err)

        self._fitted_params = p_fit
        self._fitted_param_uncertainties = p_err

        return self._fitted_params, self._fitted_param_uncertainties

    def _fit(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        initial_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        free_func: Callable[
            # TODO: correct annotation for *args?
            [Array[("num_samples",), np.float64], List[float]],
            Array[("num_samples",), np.float64],
        ],
        x_scale: Optional[float],
        y_scale: Optional[float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the fit function called from within :meth fit:, which does
        the common pre/post-processing.

        `FitBase` implementations must override this to provide a fit with appropriate
        statistics.

        :param x: x-axis data
        :param y: y-axis data
        :param initial_values: dictionary mapping model parameter names to initial
            values (either user-specified or from heuristics) to use as a starting point
            for the optimizer.
        :param bounds: dictionary mapping model parameter names to their
            `(lower, upper)` bounds. Fitted values must lie within these bounds.
        :param free_func: wrapper for the model function, taking only values for the
            fit's free parameters.
        :param x_scale: x-axis scale factor or `None` if the axis was not rescaled
        :param y_scale: y-axis scale factor or `None` if the axis was not rescaled

        :returns: tuple of dictionaries mapping model parameter names to their fitted
            values and uncertainties.
        """
        raise NotImplementedError

    def fit_significance(self) -> float:
        """Returns an estimate of the goodness of fit as a number between 0 and 1.

        This is the defined as the probability that fit residuals as large as the ones
        we observe could have arisen through chance given our assumed statistics and
        assuming that the fitted model perfectly represents the probability distribution

        A value of `1` indicates a perfect fit (all data points lie on the fitted curve)
        a value close to 0 indicates significant deviations of the dataset from the
        fitted model.
        """
        raise NotImplementedError

    def evaluate(
        self, x_fit: Optional[Union[Array[("num_samples",), np.float64], int]] = None
    ) -> Tuple[
        Array[("num_samples",), np.float64], Array[("num_samples",), np.float64]
    ]:
        """Evaluates the model function using the fitted parameter set and at x-axis
        points given by :param x_fit`:.

        :param x_fit: optional x-axis points to evaluate the model at. If `None` we use
            the dataset values. If a scalar we generate an axis of linearly spaced
            points between the minimum and maxim value of the x-axis dataset. Otherwise
            it should be an array of x-axis data points to use.

        :returns: tuple of x-axis values used and model values for those points
        """
        if self._fitted_params is None:
            raise ValueError("`fit` must be called before `evaluate`")

        x_fit = x_fit if x_fit is not None else self._x

        if np.isscalar(x_fit):
            x_fit = np.linspace(np.min(self._x), np.max(self._x), x_fit)

        y_fit = self._model.func(x_fit, self._fitted_params)
        return x_fit, y_fit

    def residuals(self) -> Array[("num_samples",), np.float64]:
        """Returns an array of fit residuals."""
        if self._y is None:
            raise ValueError("Cannot calculate residuals without a dataset")

        return self._y - self.evaluate()[0]
