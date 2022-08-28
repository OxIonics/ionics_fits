import dataclasses
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union


@dataclasses.dataclass
class FitParameter:
    """
    Arguments:
        bounds: tuple of default lower, upper bounds for the parameter.
        fixed_to: optional float specifying the value (if any) that the parameter is
            fixed to by default
        scale_func: callable which returns a scale factor giving the order-of-magnitude
            size of the parameter. This is used to improve numerics by avoiding asking
            the optimizer to work with very large or very small parameter values. The
            callable takes three arguments: the x-axis scale factor, the y-axis scale
            factor and a dictionary of fixed parameter values. If any scale func returns
            a value that is 0 or not finite (e.g. `nan`) or `None` we do not rescale the
            coordinates/parameters.
    """

    bounds: Tuple[float, float] = (-np.inf, np.inf)
    fixed_to: Optional[float] = None
    scale_func: Callable[
        [np.array, np.array, Dict[str, float]], float
    ] = lambda x_scale, y_scale, fixed_params: 1


class FitModel:
    _PARAMETERS: Dict[str, FitParameter] = {}

    @staticmethod
    def func(x: np.array, params: Dict[str, float]) -> np.array:
        """Returns the model function values at the points specified by `x` for the
        parameter values specified by `params`.
        """
        raise NotImplementedError

    @staticmethod
    def calculate_derived_params(
        fitted_params: Dict[str, float], uncertainties: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns values and uncertainties for the derived parameters based on values of
        the fitted parameters and their uncertainties.
        """
        return {}, {}

    @classmethod
    def get_parameters(cls) -> Dict[str, FitParameter]:
        """Returns a dictionary of model parameters."""
        return dict(cls._PARAMETERS)

    @staticmethod
    def estimate_parameters(x, y, known_values, bounds) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the parameter values for the specified
        dataset.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: dataset x-axis values
        :param y: dataset y-axis values
        :param known_values: dictionary of parameters whose value is known (e.g. because
            the parameter is fixed to a certain value or an estimate guess has been
            provided by the user).
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
        :param param_bounds: dictionary of tuples specifying the lower and upper bounds
            for each parameter. Entries in this dictionary override the defaults
            provided by the model. To unbound a parameter which is bounded by default
            in the model, pass `None` as bounds.
        :param fixed_params: dictionary specifying parameters which are to be held
            constant (not floated in the fit). Entries in this dictionary override the
            defaults provided by the model. To float a parameter that is fixed by
            default in the model, provide `None` as a value.
        :param initial_values: dictionary of initial parameter values. These are used
            instead of heuristics (if you find you need these it may be indicative of
            a poor heuristic so please consider filing an issue).
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
                raise ValueError(f"Invalid {kind} parameter name: {invalid}")

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

        self._fitted_params = None
        self._fitted_param_uncertainties = None

    def set_dataset(self, x, y):
        self._x = np.array(x, dtype=np.float64, copy=True)
        self._y = np.array(y, dtype=np.float64, copy=True)

        valid_pts = np.logical_and(np.isfinite(self._x), np.isfinite(self._y))
        self._x = self._x[valid_pts]
        self._y = self._y[valid_pts]

        inds = np.argsort(self._x)
        self._x = self._x[inds]
        self._y = self._y[inds]

        self._fitted_params = None
        self._fitted_param_uncertainties = None

        if self._x.shape != self._y.shape:
            raise ValueError("Shapes of x and y do not match.")

    def post_fit(self, x, y, p_fit, p_err, fixed_params, initial_values, bounds):
        """Hook called post-fit, override to implement custom functionality."""
        pass

    def fit(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Fit the dataset and return the fitted parameter values and uncertainties.
        """
        if self._x is None:
            raise ValueError("Cannot fit without first setting a dataset")

        if not self._free_params:
            raise ValueError("Attempt to fit without any floated parameters")

        x = np.array(self._x, copy=True)
        y = np.array(self._y, copy=True)

        fixed_params = dict(self._fixed_params)
        initial_values = dict(self._initial_values)
        initial_values.update(self._fixed_params)
        bounds = {
            param: np.array(bounds, copy=True)
            for param, bounds in self._param_bounds.items()
        }

        # Rescale our coordinates / parameters to avoid working with very large/small
        # values
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

        valid_scale_factors = [
            scale is not None and np.isfinite(scale) and scale != 0
            for scale in scale_factors.values()
        ]
        rescale_coords = all(valid_scale_factors)

        if rescale_coords:
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

        # Make sure we're using the known values and clip to bounds
        estimated_values = self._model.estimate_parameters(x, y, initial_values, bounds)
        estimated_values.update(initial_values)
        for param, value in estimated_values.items():
            lower, upper = bounds[param]
            estimated_values[param] = min(max(value, lower), upper)
        initial_values = estimated_values

        bounds = {
            param: bounds
            for param, bounds in bounds.items()
            if param in self._free_params
        }
        initial_values = {
            param: value
            for param, value in initial_values.items()
            if param in self._free_params
        }

        def free_func(x, *free_params):
            """Call the model function with the values of the free parameters."""
            params = {
                param: value
                for param, value in zip(self._free_params, list(free_params))
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

        self.post_fit(x, y, p_fit, p_err, fixed_params, initial_values, bounds)

        p_derived, p_derived_err = self._model.calculate_derived_params(p_fit, p_err)
        p_fit.update(p_derived)
        p_err.update(p_derived_err)
        p_fit.update({param: value for param, value in self._fixed_params.items()})
        p_err.update({param: 0 for param, _ in self._fixed_params.items()})

        self._fitted_params = p_fit
        self._fitted_param_uncertainties = p_err

        return self._fitted_params, self._fitted_param_uncertainties

    def _fit(self, x, y, initial_values, bounds, func, x_scale, y_scale):
        raise NotImplementedError

    def fit_significance(self) -> float:
        """Returns an estimate of the goodness of fit as a number between 0 and 1.

        This is the probability that the dataset could have arisen through chance under
        the assumption that the fitted model is correct and with the fit statistics. A
        value of `1` indicates a perfect fit (all data points lie on the fitted curve)
        a value close to 0 indicates significant deviations of the dataset from the
        fitted model.
        """
        raise NotImplementedError

    def evaluate(self, x_fit: Union[np.array, int] = 100):
        """Evaluates the model along x-fit and returns the tuple (x_fit, y_fit)
        with the results.

        `x_fit` may either be a scalar or an array. If it is a scalar it gives the
        number of equally spaced points between `min(self.x)` `max(self.x)`. Otherwise
        it gives the x-axis to use.
        """
        if np.isscalar(x_fit):
            x_fit = np.linspace(np.min(self._x), np.max(self._x), x_fit)
        y_fit = self._model.func(x_fit, self._fitted_params)
        return x_fit, y_fit

    def residuals(self):
        """Returns the fit residuals."""
        if self._y is None:
            raise ValueError("Cannot calculate residuals without a dataset")

        return self._y - self.evaluate(self._x)
