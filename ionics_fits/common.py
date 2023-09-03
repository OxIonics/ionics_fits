from __future__ import annotations
import dataclasses
import copy
import inspect
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from .utils import Array, ArrayLike


if TYPE_CHECKING:
    num_samples = float
    num_values = float
    num_spectrum_pts = float
    num_test_pts = float
    num_y_channels = float


logger = logging.getLogger(__name__)


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
    user_estimate: Optional[float] = None
    heuristic: Optional[float] = None
    scale_factor: float = 1
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

    def rescale(self, x_scale: float, y_scale: float, model: Model):
        """Rescales the parameter metadata based on the specified x and y data scale
        factors and returns the overall scale factor used.
        """
        scale_factor = self.scale_func(x_scale, y_scale, model)

        if scale_factor is None:
            raise ValueError("Scale factor must not be None during rescale")

        def _rescale(attr):
            if attr is None:
                return None
            return attr / scale_factor

        self.lower_bound = _rescale(self.lower_bound)
        self.upper_bound = _rescale(self.upper_bound)
        self.fixed_to = _rescale(self.fixed_to)
        self.user_estimate = _rescale(self.user_estimate)
        self.scale_factor *= scale_factor

    def get_initial_value(self) -> float:
        """
        Get initial value.

        For fixed parameters, this is the value the parameter is fixed to. For floated
        parameters, it is the value used to seed the fit. In the latter case, the
        initial value is retrieved from `user_estimate` if that attribute is not
        `None`, otherwise `heuristic` is used.
        """
        if self.fixed_to is not None:
            value = self.fixed_to
        elif self.user_estimate is not None:
            value = self.user_estimate
        elif self.heuristic is not None:
            value = self.clip(self.heuristic)
        else:
            raise ValueError("No initial value specified")

        if value < self.lower_bound or value > self.upper_bound:
            raise ValueError("Initial value outside bounds.")

        return value

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
            self.parameters = {
                name: copy.deepcopy(spec.annotations[name]) for name in spec.args[2:]
            }
        else:
            self.parameters = parameters

    def can_rescale(self, x_scale: float, y_scale: float) -> bool:
        """Returns True if the model can be rescaled"""
        return all(
            [
                param_data.can_rescale(x_scale, y_scale, self)
                for param_data in self.parameters.values()
            ]
        )

    @staticmethod
    def get_scaled_model(model, x_scale: float, y_scale: float):
        """Returns a scaled copy of a given model object

        :param model: model to be copied and rescaled
        :param x_scale: x-axis scale factor
        :param y_scale: y-axis scale factor
        :returns: a scaled copy of model
        """
        scaled_model = copy.deepcopy(model)
        for param in scaled_model.parameters.values():
            param.rescale(x_scale, y_scale, scaled_model)

        return scaled_model

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model."""
        raise NotImplementedError

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
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
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
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
        y: Array[("num_y_channels", "num_samples"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        TODO: this should act directly on self.model_parameters rather than taking
        model parameters as an argument (this is a hangover from an old design)

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        raise NotImplementedError

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
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
        :param: fitted_params: dictionary mapping model parameter names to their
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries mapping derived parameter names to their
            values and uncertainties.
        """
        return {}, {}

    def param_min_sqrs(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        parameters: Dict[str, ModelParameter],
        scanned_param: str,
        scanned_param_values: ArrayLike["num_values", np.float64],
    ) -> Tuple[float, float]:
        """Scans one model parameter while holding the others fixed to find the value
        that gives the best fit to the data (minimum sum-squared residuals).

        :param x: x-axis data
        :param y: y-axis data
        :param parameters: dictionary of model parameters. All parameters apart from the
          scanned parameter must have an initial value set.
        :param scanned_param: name of parameter to optimize
        :param scanned_param_values: array of scanned parameter values to test

        :returns: tuple with the value from :param scanned_param_values: which results
        in lowest residuals and the root-sum-squared residuals for that value.
        """
        param_values = {
            param: param_data.get_initial_value()
            for param, param_data in parameters.items()
            if param != scanned_param
        }

        scanned_param_values = np.asarray(scanned_param_values).squeeze()
        costs = np.zeros_like(scanned_param_values)
        for idx, value in np.ndenumerate(scanned_param_values):
            param_values[scanned_param] = value
            y_params = self.func(x, param_values)
            costs[idx] = np.sqrt(np.sum(np.square(y - y_params)))

        # handle a quirk of numpy indexing if only one value is passed in
        if scanned_param_values.size == 1:
            return float(scanned_param_values), float(costs)

        opt = np.argmin(costs)
        return float(scanned_param_values[opt]), float(costs[opt])

    def find_x_offset_sym_peak(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        parameters: Dict[str, ModelParameter],
        omega: Array[("num_spectrum_pts",), np.float64],
        spectrum: Array[("num_spectrum_pts",), np.float64],
        omega_cut_off: float,
        test_pts: Optional[Array[("num_test_pts",), np.float64]] = None,
        x_offset_param_name: str = "x0",
        y_offset_param_name: str = "y0",
    ):
        """Finds the x-axis offset for symmetric, peaked (maximum deviation from the
        baseline occurs at the origin) functions.

        This heuristic draws candidate x-offset points from three sources and picks the
        best one (in the least-squares residuals sense). Sources:
          - FFT shift theorem based on provided spectrum data
          - Tests all points in the top quartile of deviation from the baseline
          - Optionally, user-provided "test points", taken from another heuristic. This
            allows the developer to combine the general-purpose heuristics here with
            other heuristics which make use of more model-specific assumptions

        :param x: x-axis data
        :param y: y-axis data. For models with multiple y channels, this should contain
            data from a single channel only.
        :param parameters: dictionary of model parameters. All model parameters other
          than the x-axis offset must have initial values set before calling this method
        :param omega: FFT frequency axis
        :param spectrum: complex FFT data. For models with multiple y channels, this
          should contain data from a single channel only.
        :param omega_cut_off: highest value of omega to use in offset estimation
        :param test_pts: optional array of x-axis points to test
        :param x_offset_param_name: name of the x-axis offset model parameter
        :param y_offset_param_name: name of the y-axis offset model parameter

        :returns: an estimate of the x-axis offset
        """
        if y.ndim != 1:
            raise ValueError(
                f"{y.shape[0]} y-channels were provided to a method which takes 1."
            )

        x0_candidates = np.array([])

        if test_pts is not None:
            x0_candidates = np.append(x0_candidates, test_pts)

        try:
            fft_candidate = self.find_x_offset_fft(
                x=x, omega=omega, spectrum=spectrum, omega_cut_off=omega_cut_off
            )
            x0_candidates = np.append(x0_candidates, fft_candidate)
        except ValueError:
            pass

        y0 = parameters[y_offset_param_name].get_initial_value()
        deviations = np.argsort(np.abs(y - y0))
        top_quartile_deviations = deviations[int(len(deviations) * 3 / 4) :]
        deviations_candidates = x[top_quartile_deviations]
        x0_candidates = np.append(x0_candidates, deviations_candidates)

        best_x0, _ = self.param_min_sqrs(
            x=x,
            y=y,
            parameters=parameters,
            scanned_param=x_offset_param_name,
            scanned_param_values=x0_candidates,
        )
        return best_x0

    def find_x_offset_sampling(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        parameters: Dict[str, ModelParameter],
        width: float,
        x_offset_param_name: str = "x0",
    ) -> float:
        """Finds the x-axis offset of a dataset by stepping through a range of potential
        offset values and picking the one that gives the lowest residuals.

        This method is typically called during parameter estimation after all other
        model parameters have been estimated from the periodogram (which is not itself
        sensitive to offsets).

        There are a few ways one can implemented this functionality: for strongly
        peaked functions we could have used a simple peak search; we could have used an
        FFT, fitted a line to the phase and used the Fourier Transform Shift Theorem.

        This function takes a more brute-force approach by evaluating the model at a
        range of offset values, picking the one that gives the lowest residuals. This
        may be appropriate where one needs the estimate to be highly robust in the face
        of noisy, irregularly sampled data.

        :param x: x-axis data
        :param y: y-axis data
        :param parameters: dictionary of model parameters.
        :param width: width of the feature we're trying to find (e.g. FWHMH). Used to
            pick the spacing between offset values to try.
        :param x_offset_param_name: name of the x-axis offset parameter

        :returns: an estimate of the x-axis offset
        """
        offsets = np.arange(min(x), max(x), width / 6)
        return self.param_min_sqrs(x, y, parameters, x_offset_param_name, offsets)[0]

    def find_x_offset_fft(
        self,
        x: Array[("num_samples",), np.float64],
        omega: Array[("num_spectrum_pts",), np.float64],
        spectrum: Array[("num_spectrum_pts",), np.float64],
        omega_cut_off: float,
    ) -> float:
        """Finds the x-axis offset of a dataset from the phase of an FFT.

        This method is typically called during parameter estimation after all other
        model parameters have been estimated from the periodogram (which is not itself
        sensitive to offsets).

        This method uses the FFT shift theorem to extract the offset from the phase
        slope of an FFT. At present it only supports models with a single y channel.

        :param omega: FFT frequency axis
        :param spectrum: complex FFT data. For models with multiple y channels, this
          should contain data from a single channel only.
        :param omega_cut_off: highest value of omega to use in offset estimation

        :returns: an estimate of the x-axis offset
        """
        if spectrum.ndim != 1:
            raise ValueError(
                f"{spectrum.shape[1]} y channels were provided to a method which "
                "takes 1"
            )

        keep = omega < omega_cut_off
        if np.sum(keep) < 2:
            raise ValueError("Insufficient data below cut-off")

        omega = omega[keep]
        phi = np.unwrap(np.angle(spectrum[keep]))
        phi -= phi[0]

        p = np.polyfit(omega, phi, deg=1)

        x0 = min(x) - p[0]
        x0 = x0 if x0 > min(x) else x0 + x.ptp()
        return x0

    def get_initial_values(
        self, model_parameters: Optional[Dict[str, ModelParameter]] = None
    ) -> Dict[str, float]:
        """Returns a dictionary mapping model parameter names to their initial values.

        :param model_parameters: optional dictionary mapping model parameter names to
           :class ModelParameter:s
        """
        model_paramers = model_parameters or self.parameters
        return {
            param: param_data.get_initial_value()
            for param, param_data in model_paramers.items()
        }


class Fitter:
    """Base class for fitters.

    Fitters perform maximum likelihood parameter estimation on a dataset under the
    assumption of a certain model and statistics (normal, binomial, etc) and store the
    results as attributes.

    Attributes:
        x: 1D ndarray of shape (num_samples,) containing x-axis values of valid points.
        y: 2D ndarray of shape (num_y_channels, num_samples) containing y-axis values
            of valid points.
        sigma: optional 2D ndarray of shape (num_y_channels, num_samples) containing
            y-axis standard deviations of valid points. Only used by `NormalFitter`.
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
        fit_significance: if applicable, the fit significance as a number between 0 and
            1 (1 indicates perfect agreement between the fitted model and input
            dataset). See :meth _fit_significance: for details.
        free_parameters: list of names of the model parameters floated during the fit
    """

    x: Array[("num_samples",), np.float64]
    y: Array[("num_y_channels", "num_samples"), np.float64]
    sigma: Optional[Array[("num_y_channels", "num_samples"), np.float64]]
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    derived_values: Dict[str, float]
    derived_uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model
    fit_significance: Optional[float]
    free_parameters: List[str]

    def __init__(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_y_channels", "num_samples"), np.float64],
        model: Model,
        sigma: Optional[
            ArrayLike[("num_y_channels", "num_samples"), np.float64]
        ] = None,
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

        # Sanitize input dataset
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True)
        sigma = None if sigma is None else np.array(sigma, dtype=np.float64, copy=True)

        if x.ndim != 1:
            raise ValueError("x-axis data must be a 1D array.")

        if y.ndim > 2:
            raise ValueError("y-axis data must be a 1D or 2D array.")

        # Ensure a common shape for all data related to y-axis, regardless of
        # user input
        y = np.atleast_2d(y)
        if sigma is not None:
            sigma = np.atleast_2d(sigma)

        if x.shape[0] != y.shape[1]:
            raise ValueError(
                "Number of x-axis and y-axis samples must match "
                f"(got {x.shape} and {y.shape})."
            )
        if y.shape[0] != model.get_num_y_channels():
            raise ValueError(
                f"Expected {model.get_num_y_channels()} y-channels, got {y.shape[1]}."
            )

        if sigma is not None and sigma.shape != y.shape:
            raise ValueError(
                f"Shapes of sigma (got {sigma.shape}) and y (got {y.shape}) must match."
            )

        if sigma is not None and not np.all(sigma != 0):
            logger.warning("Ignoring points with zero uncertainty.")

        valid_x = np.isfinite(x)
        valid_y = np.isfinite(y)
        valid_sigma = (
            None if sigma is None else np.logical_and(np.isfinite(sigma), sigma != 0)
        )

        valid_y = np.all(valid_y, axis=0)
        valid_sigma = None if sigma is None else np.all(valid_sigma, axis=0)

        valid_pts = np.logical_and(valid_x, valid_y)
        if sigma is not None:
            valid_pts = np.logical_and(valid_pts, valid_sigma)
        valid_pts = valid_pts.squeeze()

        x = x[valid_pts]
        y = y[:, valid_pts]
        sigma = None if sigma is None else sigma[:, valid_pts]

        inds = np.argsort(x)
        x = x[inds]
        y = y[:, inds]
        sigma = None if sigma is None else sigma[:, inds]

        self.x = x
        self.y = y
        self.sigma = sigma

        # Rescale coordinates to improve numerics (optimizers need to do things like
        # calculate numerical derivatives which is easiest if x and y are O(1)).
        #
        # Currently we use a single scale factor for all y channels. This may change in
        # future
        x_scale = np.max(np.abs(x))
        y_scale = np.max(np.abs(y))

        if model.can_rescale(x_scale, y_scale):
            scaled_model = model.get_scaled_model(model, x_scale, y_scale)
        else:
            x_scale = 1
            y_scale = 1
            scaled_model = copy.deepcopy(model)

        x = x / x_scale
        y = y / y_scale
        sigma = None if sigma is None else sigma / y_scale

        scaled_model.estimate_parameters(x, y, scaled_model.parameters)

        for param, param_data in scaled_model.parameters.items():
            try:
                param_data.get_initial_value()
            except ValueError:
                raise RuntimeError(
                    "No fixed_to, user_estimate or heuristic specified"
                    f" for parameter `{param}`."
                )

        self.fixed_parameters = {
            param_name: param_data.fixed_to
            for param_name, param_data in scaled_model.parameters.items()
            if param_data.fixed_to is not None
        }
        self.free_parameters = [
            param_name
            for param_name, param_data in scaled_model.parameters.items()
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
            y = scaled_model.func(x, params)
            return y

        fitted_params, uncertainties = self._fit(
            x, y, sigma, scaled_model.parameters, free_func
        )
        fitted_params.update(
            {param: value for param, value in self.fixed_parameters.items()}
        )
        uncertainties.update({param: 0 for param in self.fixed_parameters.keys()})

        fitted_params = {
            param: value * scaled_model.parameters[param].scale_factor
            for param, value in fitted_params.items()
        }
        uncertainties = {
            param: value * scaled_model.parameters[param].scale_factor
            for param, value in uncertainties.items()
        }

        initial_values = {
            param: param_data.get_initial_value()
            * scaled_model.parameters[param].scale_factor
            for param, param_data in scaled_model.parameters.items()
        }

        (
            self.derived_values,
            self.derived_uncertainties,
        ) = model.calculate_derived_params(self.x, self.y, fitted_params, uncertainties)

        self.model = model
        self.values = fitted_params
        self.uncertainties = uncertainties
        self.initial_values = initial_values
        self.fit_significance = self._fit_significance()

    @staticmethod
    def _fit(
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        sigma: Optional[Array[("num_y_channels", "num_samples"), np.float64]],
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., Array[("num_y_channels", "num_samples"), np.float64]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the parameter estimation.

        `Fitter` implementations must override this method to provide a fit with
        appropriate statistics.

        :param x: rescaled x-axis data, must be a 1D array
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
        self,
        transpose_and_squeeze=False,
        x_fit: Optional[Array[("num_samples",), np.float64]] = None,
    ) -> Tuple[
        Array[("num_samples",), np.float64],
        Array[("num_y_channels", "num_samples"), np.float64],
    ]:
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

    def residuals(self) -> Array[("num_y_channels", "num_samples"), np.float64]:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]
