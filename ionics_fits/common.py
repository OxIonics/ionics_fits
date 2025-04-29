from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import TSCALE_FUN, TX_SCALE, TY_SCALE, Array, ArrayLike, scale_undefined

if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_x_axes = float
    num_y_axes = float


logger = logging.getLogger(__name__)


TX = Union[
    float,
    ArrayLike[("num_samples",), np.float64],
    ArrayLike[
        (
            "num_x_axes",
            "num_samples",
        ),
        np.float64,
    ],
]
TY = Union[
    float,
    Array[("num_samples"), np.float64],
    Array[("num_y_axes", "num_samples"), np.float64],
]


@dataclasses.dataclass
class ModelParameter:
    """Represents a model parameter.

    Attributes:
        scale_func: callable returning a scale factor which the parameter must be
            *multiplied* by if it was fitted using ``x`` and ``y`` data that has been
            *multiplied* by the given scale factors. Scale factors are used to improve
            numerical stability by avoiding asking the optimizer to work with very large
            or very small values of ``x`` and ``y``. The callable takes the x-axis and
            y-axis scale factors as arguments. A number of default scale functions are
            provided for convenience in :py:mod:`ionics_fits.utils`.
        lower_bound: lower bound for the parameter. Fitted values are guaranteed to be
            greater than or equal to the lower bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        upper_bound: upper bound for the parameter. Fitted values are guaranteed to be
            lower than or equal to the upper bound. Parameter bounds may be used by
            fit heuristics to help find good starting points for the optimizer.
        fixed_to: if not ``None``, the model parameter is fixed to this value during
            fitting instead of being floated. This value may additionally be used by
            the heuristics to help find good initial values for other model parameters.
            The value of ``fixed_to`` must lie within the bounds of the parameter.
        user_estimate: if not ``None`` and the parameter is not fixed, this value is
            used as an initial value during fitting rather than obtaining a value from
            the heuristics. This value may additionally be used by the heuristics to
            help find good initial values for other model parameters. The value of
            ``user_estimate`` must lie within the bounds of the parameter.
        heuristic: if both of ``fixed_to`` and ``user_estimate`` are ``None``, this
            value is used as an initial value during fitting. It is set by the
            :class:`~ionics_fits.common.Model`\' s
            :meth:`~ionics_fits.common.Model.estimate_parameters` method  and should not
            be set by the user.
    """

    scale_func: TSCALE_FUN
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    fixed_to: Optional[float] = None
    user_estimate: Optional[float] = None
    heuristic: Optional[float] = None
    scale_factor: Optional[float] = dataclasses.field(init=False, default=None)

    _metadata_attrs: List[str] = dataclasses.field(init=False)

    def __post_init__(self):
        self._metadata_attrs = [
            "lower_bound",
            "upper_bound",
            "fixed_to",
            "user_estimate",
            "heuristic",
        ]

    def _format_metadata(self) -> List[str]:
        data = []

        if self.lower_bound != -np.inf:
            data.append(f"lower_bound={self.lower_bound}")
        if self.upper_bound != np.inf:
            data.append(f"upper_bound={self.upper_bound}")
        if self.fixed_to is not None:
            data.append(f"fixed_to={self.fixed_to}")
        if self.user_estimate is not None:
            data.append(f"user_estimate={self.user_estimate}")
        data.append(f"scale_func={self.scale_func.__name__}")

        return data

    def __repr__(self):
        return f"<{self.__class__.__name__}(" + ",".join(self._format_metadata()) + ")>"

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name in ["scale_factor", "_metadata_attrs"]:
            return attr

        scale_factor = self.scale_factor
        if attr is None or scale_factor is None:
            return attr

        if name in self._metadata_attrs:
            attr /= scale_factor

        return attr

    def __setattr__(self, name, value):
        scale_factor = self.scale_factor

        if None not in [scale_factor, value] and name in self._metadata_attrs:
            value *= scale_factor
        object.__setattr__(self, name, value)

    def rescale(self, x_scales: TX_SCALE, y_scales: TY_SCALE):
        r"""Rescales the parameter metadata based on the specified x and y data scale
        factors.

        Rescaling affects the values of :attr:`lower_bound`\, :attr:`upper_bound`\,
        :attr:`fixed_to`\, :attr:`user_estimate`\, and :attr:`heuristic`.

        :param x_scales: array of x-axis scale factors
        :param y_scales: array of y-axis scale factors
        """
        if self.scale_factor is not None:
            raise RuntimeError("Attempt to rescale an already rescaled model parameter")
        self.scale_factor = self.scale_func(x_scales, y_scales)

    def unscale(self):
        """Disables rescaling of the parameter metadata"""
        if self.scale_factor is None:
            raise RuntimeError(
                "Attempt to unscale a model parameter which was not rescaled."
            )
        self.scale_factor = None

    def get_initial_value(self, default: Optional[float] = None) -> float:
        """Returns the parameter's initial value.

        For fixed parameters, this is the value the parameter is fixed to. For floated
        parameters, it is the value used to seed the fit. In the latter case, the
        initial value is retrieved from the :attr:`user_estimate` if available,
        otherwise the :attr:`heuristic` is used.

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
        Returns ``True`` if the parameter is fixed, has a user estimate or a heuristic.
        """
        values = [self.fixed_to, self.user_estimate, self.heuristic]
        return any([None is not value for value in values])

    def has_user_initial_value(self) -> bool:
        """Returns ``True`` if the parameter is fixed or has a user estimate"""
        return self.fixed_to is not None or self.user_estimate is not None

    def clip(self, value: float) -> float:
        """Clips a value to lie between the parameter's lower and upper bounds.

        :param value: value to be clipped
        :returns: clipped value
        """
        return np.clip(value, self.lower_bound, self.upper_bound)


class Model:
    """Base class for fit models.

    A model groups a function to be fitted with associated metadata (parameter names,
    default bounds etc) and heuristics. It is agnostic about the method of fitting or
    the data statistics.

    Models may be used either as part of a fit (see :class:`~ionics_fits.common.Fitter`)
    or as a standalone function (see :meth:`__call__`).

    Class Attributes
    ================

    Attributes:
        parameters: dictionary mapping parameter names to
            :class:`~ionics_fits.common.ModelParameter` s. The parameters may be
            modified to alter their properties (bounds, user estimates, etc.).
        internal_parameters: list of "internal" model parameters. These are not directly
            used during the fit, but are rescaled in the same way as regular model
            parameters. These are used, for example, by
            :py:mod:`~ionics_fits.models.transformations` models.
    """

    parameters: Dict[str, ModelParameter]
    internal_parameters: List[ModelParameter]

    def __init__(
        self,
        parameters: Optional[Dict[str, ModelParameter]] = None,
        internal_parameters: Optional[List[ModelParameter]] = None,
    ):
        r"""
        :param parameters: optional dictionary mapping parameter names to
            :class:`~ionics_fits.common.ModelParameter`\ s. This should be ``None``
            (default) if the model has a static set of parameters, in which case the
            parameter dictionary is generated from the call signature of
            :func:`~ionics_fits.common.Model._func`. The model parameters are stored as
            :attr:`~ionics_fits.common.Model.parameters` and may be modified
            after construction to change the model behaviour during fitting (e.g. to
            change the bounds, fixed parameters, etc).
        :param internal_parameters: optional list of "internal" model parameters, which
            are not exposed to the user as arguments of
            :func:`~ionics_fits.common.Model.func`. Internal parameters
            are rescaled in the same way as regular model parameters, but are not
            otherwise used by :class:`~ionics_fits.common.Model`. These are used by
            :py:mod:`~ionics_fits.models.transformations` models, which modify the
            behaviour of other models.
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

    def __call__(self, x: TX, transpose_and_squeeze=False, **kwargs: float) -> TY:
        r"""Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the results.

        Example:

        .. testcode::

            import numpy as np
            from matplotlib import pyplot as plt
            from ionics_fits.models.sinusoid import Sinusoid

            model = Sinusoid()
            x = np.linspace(0, 1)
            y = model(x, True, a=1, omega=2*np.pi, phi=0, y0=0)
            plt.plot(x, y)

        :param x: x-axis data. For models with more than one x-axis, the data should
            be shaped ``(num_x_axes, num_samples)``. For models with a single x-axis
            dimension, a 1D array may be used instead.
        :param transpose_and_squeeze: if True, the results arrays are transposed and
            squeezed proior to being returned. This is intended to be used for plotting,
            since matplotlib requires different y-series to be stored as columns.
        :param \**kwargs: values for model parameters. All model parameters
            which are not :attr:`~ionics_fits.common.ModelParameter.fixed_to` a value
            must be specified. Any parameters which are not specified default to their
            fixed values.
        :returns: the model function values
        """
        args = {
            param_name: param_data.fixed_to
            for param_name, param_data in self.parameters.items()
            if param_data.fixed_to is not None
        }
        args.update(kwargs)
        y = self.func(x, args)

        if transpose_and_squeeze:
            y = y.T.squeeze()  # pytype: disable=attribute-error

        return y

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        """
        Returns a tuple of lists of bools specifying whether the model can be rescaled
        along each x- and y-axes dimension.
        """
        raise NotImplementedError

    def rescale(self, x_scales: TX_SCALE, y_scales: TY_SCALE):
        """Rescales the model parameters based on the specified x and y data scale
        factors.

        All :attr:`~ionics_fits.common.Model.parameters` and
        :attr:`~ionics_fits.common.Model.internal_parameters` are rescaled.

        :param x_scales: array of x-axis scale factors
        :param y_scales: array of y-axis scale factors
        """
        for param_name, param_data in self.parameters.items():
            if param_data.scale_func == scale_undefined:
                raise RuntimeError(
                    f"Parameter {param_name} has an undefined scale function"
                )
            try:
                param_data.rescale(x_scales, y_scales)
            except Exception as e:
                raise RuntimeError(f"Error rescaling parameter {param_name}") from e

        for param_data in self.internal_parameters:
            param_data.rescale(x_scales, y_scales)

    def unscale(self):
        """Disables rescaling of the model parameters."""
        parameters = list(self.parameters.values()) + self.internal_parameters
        for param_data in parameters:
            param_data.unscale()

    def get_num_x_axes(self) -> int:
        """Returns the number of x-axis dimensions the model has."""
        raise NotImplementedError

    def get_num_y_axes(self) -> int:
        """Returns the number of y-axis dimensions the model has."""
        raise NotImplementedError

    def clear_heuristics(self):
        r"""Clear the heuristics for all model parameters (both exposed and internal).

        This is mainly used in :py:mod:`~ionics_fits.models.transformations`\-type
        models where the parameter estimator my be run multiple times for the same model
        instance.
        """
        for param_data in self.parameters.values():
            param_data.heuristic = None
        for param_data in self.internal_parameters:
            param_data = None

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        To use the model as a function outside of a fit,
        :meth:`~ionics_fits.common.Model.__call__` generally
        provides a more convenient interface.

        Overload this to provide a model function with a dynamic set of parameters,
        otherwise prefer to override :func:`~ionics_fits.common.Model._func`.

        :param x: x-axis data
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        x = np.atleast_2d(x)
        return self._func(x, **param_values)

    def _func(self, x: TX) -> TY:
        r"""Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the results.

        Overload this in preference to :func:`~ionics_fits.common.Model.func` unless the
        :class:`~ionics_fits.common.Model` takes a dynamic set of parameters.

        :class:`~ionics_fits.common.ModelParameter`\ s should be used as the type
        annotations for the parameters arguments to define the model's parameters.
        These are used in the construction to generate the
        :attr:`~ionics_fits.common.Model.parameters` dictionary::

            from ionics_fits.utils import scale_x, scale_y

            def _func(
                self,
                x,
                a: ModelParameter(lower_bound=-1., scale_func=scale_y()),
                x0: ModelParameter(fixed_to=0, scale_func=scale_x())
            ):
                ...

        :param x: x-axis data
        :returns: array of model values
        """
        raise NotImplementedError

    def estimate_parameters(self, x: TX, y: TY):
        """Set heuristic values for model parameters.

        Typically called by :class:`~ionics_fits.common.Fitter`.

        Implementations of this method must ensure that all parameters have an initial
        value set (at least one of :attr:`~ionics_fits.common.ModelParameter.fixed_to`,
        :attr:`~ionics_fits.common.ModelParameter.user_estimate` or
        :attr:`~ionics_fits.common.ModelParameter.heuristic` must not be ``None`` for
        each parameter).

        Implementations should aim to make use of all information supplied by the user
        (bounds, user estimates, fixed values) to provide the best initial guesses for
        all parameters.

        The x and y data is sorted along the x-axis dimensions and is filtered to remove
        points with non-finite x or y values and are is rescaled if supported by the
        model.

        :param x: x-axis data
        :param y: y-axis data
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
            fitted values.
        :param fit_uncertainties: dictionary mapping model parameter names to
            their fit uncertainties.
        :returns: tuple of dictionaries containing the derived parameter values and
            uncertainties.
        """
        return {}, {}


class Fitter:
    r"""Base class for fitters.

    Fitters perform maximum likelihood parameter estimation on a dataset under the
    assumption of a certain model and statistics (normal, binomial, etc) and store the
    results as attributes.

    For details about the various fitter subclasses provided by ``ionics_fits``\ , see
    :ref:`fitters`.

    Usage
    =====

    Basic usage::

        import numpy as np
        from matplotlib import pyplot as plt

        from ionics_fits.models.polynomial import Line
        from ionics_fits.normal import NormalFitter

        a = 3.2
        y0 = -9

        x = np.linspace(-10, 10)
        y = a * x + y0

        fit = NormalFitter(x, y, model=Line())
        print(f"Fitted: y = {fit.values['a']:.3f} * x + {fit.values['y0']:.3f}")

        plt.plot(x, y)
        plt.plot(*fit.evaluate())
        plt.show()

    The fit may be configured by modifying the :class:`~ionics_fits.common.Model`\ 's
    :attr:`~ionics_fits.common.Model.parameters` dictionary. This allows one to:

      * set upper and lower bounds for each parameter
      * control which parameters are fixed / floated
      * provide user estimates to be used instead of the
        :class:`~ionics_fits.common.Model`\ 's heuristics

    As an example, let's fit a sinusoid, whose frequency is already known:

    .. testcode::

        import numpy as np
        from matplotlib import pyplot as plt

        from ionics_fits.models.sinusoid import Sinusoid
        from ionics_fits.normal import NormalFitter

        omega = 2 * np.pi
        model = Sinusoid()
        model.parameters["omega"].fixed_to = omega

        params = {
            "a": 2,
            "omega": omega,
            "phi": 0.5 * np.pi,
            "y0": 0,
            "x0": 0,
            "tau": np.inf,
        }

        x = np.linspace(-3, 3, 100)
        y = model(x, True, **params)

        fit = NormalFitter(x, y, model=model)
        print(f"Amplitude: dataset = {params['a']:.3f}, fit = {fit.values['a']:.3f}")
        print(f"Phase: dataset = {params['phi']:.3f}, fit = {fit.values['phi']:.3f}")

        plt.plot(*fit.evaluate(None, True), '-.o', label="fit")
        plt.plot(x, y, label="data")
        plt.grid()
        plt.legend()
        plt.show()

    .. testoutput::

        Amplitude: dataset = 2.000, fit = 2.000
        Phase: dataset = 1.571, fit = 1.571

    ``ionics_fits`` supports fitting datasets with arbitrary x-axis and y-axis
    dimensions. The :py:mod:`~ionics_fits.models.transformations` module provides a
    number of classes which allow higher-dimensional models to be constructed from
    lower-dimension models.

    Example of fitting a dataset with a 2D x-axis:

    .. testcode::

        import numpy as np
        from matplotlib import pyplot as plt

        from ionics_fits.models.multi_x import Gaussian2D
        from ionics_fits.normal import NormalFitter

        omega = 2 * np.pi  # we know the frequency
        model = Gaussian2D()

        params = {
            "x0_x0": 0,
            "x0_x1": 3,
            "sigma_x0": 2,
            "sigma_x1": 3,
            "y0": 0,
            "a": 9,
        }

        x_0_ax = np.linspace(-3, 3, 50)
        x_1_ax = np.linspace(-10, 10, 100)
        x_0_mesh, x_1_mesh = np.meshgrid(x_0_ax, x_1_ax)
        x_shape = x_0_mesh.shape

        x = np.vstack((x_0_mesh.ravel(), x_1_mesh.ravel()))
        y = model(x, **params)

        fit = NormalFitter(x, y, model=model)

        _, y_fit = fit.evaluate(x)
        y_fit = y_fit.reshape(x_shape)

        _, axs = plt.subplots(2, 1)
        axs[0].pcolormesh(x_0_mesh, x_1_mesh, y.reshape(x_shape))
        axs[1].pcolormesh(x_0_mesh, x_1_mesh, y_fit)
        axs[0].title.set_text("Model")
        axs[1].title.set_text("Fit")
        axs[0].grid()
        axs[1].grid()
        plt.show()

    As an example of fitting a dataset with a 2D y-axis, here's how to fit Rabi flopping
    on a pair of qubits. We'll assume all parameters are the same for the two qubits,
    other than the Rabi frequencies:

    .. testcode::

        import pprint
        import numpy as np
        from matplotlib import pyplot as plt

        from ionics_fits.models.rabi import RabiFlopTime
        from ionics_fits.models.transformations.repeated_model import RepeatedModel
        from ionics_fits.normal import NormalFitter

        params = {
            "omega_0": 2 * np.pi * 1,
            "omega_1": 2 * np.pi * 2,
            "delta": 0,
            "P_readout_e": 1,
            "P_readout_g": 0
        }

        rabi_model = RabiFlopTime(start_excited=False)
        model = RepeatedModel(
            model=rabi_model,
            num_repetitions=2,
            common_params=[
                param_name for param_name in rabi_model.parameters.keys()
                if param_name != "omega"
            ]
        )

        t = np.linspace(0, 3, 100)
        y = model(t, **params)

        fit = NormalFitter(t, y, model)

        pprint.pprint(fit.values)

        plt.plot(t, y.T, ".")
        plt.legend(("qubit 0", "qubit 1"))
        plt.gca().set_prop_cycle(None)
        plt.plot(*fit.evaluate(None, True))
        plt.grid()
        plt.show()

    .. testoutput::

        {'P_readout_e': np.float64(0.9999999999),
         'P_readout_g': np.float64(1e-10),
         'delta': np.float64(0.0),
         'omega_0': np.float64(6.283185307179586),
         'omega_1': np.float64(12.56637061436225),
         't_dead': np.float64(0.0),
         'tau': np.float64(inf)}

    Class Attributes
    ================

    Attributes:
        x: x-axis data. The input data is sorted along the x-axis dimensions and
            filtered to contain only the "valid" points where x and y are finite.
        y: y-axis data. The input data is sorted along the x-axis dimensions x and
            filtered to contain only the "valid" points where x and y are finite.
        sigma: standard errors for each point. This is stored as an array with the same
            shape as `y`.
        values: dictionary mapping model parameter names to their fitted values
        uncertainties: dictionary mapping model parameter names to their fit
            uncertainties. For sufficiently large datasets, well-formed problems and
            ignoring covariances these are the 1-sigma confidence intervals (roughly:
            there is a 1/3 chance that the true parameter values differ from their
            fitted values by more than this much). These uncertainties are generally
            only useful when the ``covariances`` are small.
        derived_values: dictionary mapping names of derived parameters (parameters which
            are not part of the fit, but are calculated by the model from the fitted
            parameter values) to their values
        derived_uncertainties: dictionary mapping names of derived parameters to their
            fit uncertainties
        initial_values: dictionary mapping model parameter names to the initial values
            used to seed the fit.
        model: the fit model
        covariances: matrix of covariances between the floated parameters. The ordering
            of parameters in the covariance matrix matches ``free_parameters``. The
            fit uncertainties are calculated as the square root of the diagonals of
            the covariance matrix.
        free_parameters: list of names of the model parameters floated during the fit
        x_scales: the applied x-axis scale factors
        y_scales: the applied y-axis scale factors
    """

    x: TX
    y: TY
    sigma: Optional[TY]
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    derived_values: Dict[str, float]
    derived_uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model
    covariances: Array[("num_free_params", "num_free_params"), np.float64]
    free_parameters: List[str]
    x_scales: TX_SCALE
    y_scales: TY_SCALE

    def __init__(self, x: TX, y: TY, model: Model):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data. For models with more than one x-axis dimension, ``x``
            should be in the form ``(num_x_axes, num_samples)``.
        :param y: y-axis data.For models with more than one y-axis dimension, ``y``
            should be in the form ``(num_y_axes, num_samples)``.
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults. The
            model is (deep) copied and stored as an attribute.
        """
        self.model = copy.deepcopy(model)

        self.x = np.atleast_2d(np.array(x, dtype=np.float64, copy=True))
        self.y = np.atleast_2d(np.array(y, dtype=np.float64, copy=True))

        self.sigma = self.calc_sigma()
        self.sigma = np.atleast_2d(self.sigma) if self.sigma is not None else None

        if self.x.ndim != 2:
            raise ValueError("x-axis data must be a 1D or 2D array.")

        if self.y.ndim != 2:
            raise ValueError("y-axis data must be a 1D or 2D array.")

        if self.x.shape[1] != self.y.shape[1]:
            raise ValueError(
                "Number of samples in the x and y datasets must match "
                f"(got {self.x.shape[1]} along x and {self.y.shape[1]} along y)."
            )

        if self.x.shape[0] != self.model.get_num_x_axes():
            raise ValueError(
                f"Expected {self.model.get_num_x_axes()} x axes, "
                f"got {self.x.shape[0]}."
            )

        if self.y.shape[0] != self.model.get_num_y_axes():
            raise ValueError(
                f"Expected {self.model.get_num_y_axes()} y axes, "
                f"got {self.y.shape[0]}."
            )

        if self.sigma is not None and self.sigma.shape != self.y.shape:
            raise ValueError(
                f"Shapes of sigma and y must match (got {self.sigma.shape} and "
                f"{self.y.shape})."
            )

        valid_x = np.all(np.isfinite(self.x), axis=0)
        valid_y = np.all(np.isfinite(self.y), axis=0)
        valid_pts = np.logical_and(valid_x, valid_y)
        assert valid_pts.ndim == 1

        (valid_inds,) = np.where(valid_pts)
        sorted_inds = valid_inds[np.lexsort(np.flipud(self.x[:, valid_inds]))]

        self.x = self.x[:, sorted_inds]
        self.y = self.y[:, sorted_inds]

        if self.sigma is not None:
            self.sigma = self.sigma[:, sorted_inds]
            if np.any(self.sigma == 0) or not np.all(np.isfinite(self.sigma)):
                raise RuntimeError(
                    "Dataset contains points with zero or infinite uncertainty."
                )

        # Rescale coordinates to improve numerics (optimizers need to do things like
        # calculate numerical derivatives which is easiest if x and y are O(1)).
        rescale_xs, rescale_ys = self.model.can_rescale()

        if len(rescale_xs) != self.model.get_num_x_axes():
            raise ValueError(
                "Unexpected number of x-axis results returned from model.can_rescale: "
                f"expected {self.model.get_num_x_axes()}, got {len(rescale_xs)}"
            )

        if len(rescale_ys) != self.model.get_num_y_axes():
            raise ValueError(
                "Unexpected number of y-axis results returned from model.can_rescale: "
                f"expected {self.model.get_num_y_axes()}, got {len(rescale_ys)}"
            )

        self.x_scales = np.array(
            [
                max(np.abs(self.x[idx, :])) if rescale else 1.0
                for idx, rescale in enumerate(rescale_xs)
            ]
        )

        self.y_scales = np.array(
            [
                max(np.abs(self.y[idx, :])) if rescale else 1.0
                for idx, rescale in enumerate(rescale_ys)
            ]
        )

        # Corner-case if a y-axis dimension has values that are all 0
        self.y_scales = np.array(
            [
                y_scale if (y_scale != 0 and np.isfinite(y_scale)) else 1.0
                for y_scale in self.y_scales
            ]
        )

        self.model.rescale(self.x_scales, self.y_scales)

        x_scaled = self.x / self.x_scales[:, None]
        y_scaled = self.y / self.y_scales[:, None]

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

        def free_func(x: TX, *free_param_values: float) -> TY:
            """Call the model function with the values of the free parameters."""
            params = {
                param: value
                for param, value in zip(self.free_parameters, list(free_param_values))
            }
            params.update(self.fixed_parameters)
            return self.model.func(x, params)

        fitted_params, uncertainties, covariances = self._fit(
            x_scaled, y_scaled, self.model.parameters, free_func
        )
        fitted_params.update(
            {param: value for param, value in self.fixed_parameters.items()}
        )
        uncertainties.update({param: 0 for param in self.fixed_parameters.keys()})

        # Make sure final values lie within parameter bounds
        # e.g. for periodic parameters we want to make sure the final value lies within
        # the specified range
        fitted_params = {
            param: self.model.parameters[param].clip(value)
            for param, value in fitted_params.items()
        }

        self.values = {
            param: value * self.model.parameters[param].scale_factor
            for param, value in fitted_params.items()
        }
        self.uncertainties = {
            param: value * self.model.parameters[param].scale_factor
            for param, value in uncertainties.items()
        }

        # rescale the covariance matrix
        free_param_scales = np.array(
            [
                self.model.parameters[param].scale_factor
                for param in self.free_parameters
            ]
        )
        covariance_scales = np.tile(free_param_scales, (len(self.free_parameters), 1))
        covariance_scales = np.multiply(covariance_scales, covariance_scales.T)
        self.covariances = np.multiply(covariances, covariance_scales)

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
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Array[("num_free_params", "num_free_params"), np.float64],
    ]:
        """Implementation of the parameter estimation.

        ``Fitter`` implementations must override this method to provide a fit with
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
        x_fit: Optional[TX] = None,
        transpose_and_squeeze=False,
    ) -> Tuple[TX, TY]:
        r"""Evaluates the model function using the fitted parameter set.

        :param x_fit: optional x-axis points to evaluate the model at. If ``None``\ , we
            use the stored value of :attr:`x`.
        :param transpose_and_squeeze: if ``True``, the results arrays are transposed and
            squeezed prior to being returned. This is intended to be used for plotting,
            since matplotlib requires different y-series to be stored as columns.

        :returns: tuple of x-axis values used and corresponding y-axis found by
            evaluating the model.
        """
        x_fit = np.atleast_2d(x_fit if x_fit is not None else self.x)
        y_fit = np.atleast_2d(self.model.func(x_fit, self.values))

        if transpose_and_squeeze:
            return x_fit.T.squeeze(), y_fit.T.squeeze()
        return x_fit, y_fit

    def residuals(self) -> TY:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]

    def calc_sigma(self) -> Optional[TY]:
        """Returns an array of standard error values for each y-axis data point

        Subclasses must override this.
        """
        raise NotImplementedError
