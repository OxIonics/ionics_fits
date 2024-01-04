import copy
import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np


from .. import Fitter, Model, NormalFitter
from ..models import RepeatedModel
from ..utils import Array, ArrayLike


if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_samples_x_flat = float
    num_samples_ax_0 = float
    num_samples_ax_1 = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


TX2D = Tuple[
    ArrayLike[("num_samples_ax_0",), np.float64],
    ArrayLike[("num_samples_ax_1",), np.float64],
]
# NB ordering of x-axis samples here is chosen to play nicely with np.meshgrid
TY2D = Union[
    ArrayLike[("num_y_channels", "num_samples_ax_0", "num_samples_ax_1"), np.float64],
    ArrayLike[("num_samples_ax_0", "num_samples_ax_1"), np.float64],
]

logger = logging.getLogger(__name__)


class Model2D:
    """Base class providing a :class Model:-like interface for models with
    2-dimensional x-axes.

    2D x-axis data is represented by the tuple `x=(x_axis_0, x_axis_1)`.

    This class provides a means of combining a pair of 1D :class Model:s to create a 2D
    model. Each 1D model is a function of one x-axis dimension:
      `model_0 = models[0] = f(x_axis_0)`
      `model_1 = models[1] = g(x_axis_1)`

    All y-axis data is generated from the output of the first model; the output from
    the second model provides the values of certain "result" parameters used by the
    first model. In other words:
      `y(x_0, x_1) = model_0(x_0 | result_params = model_1(x_1))`

    - All parameters from the two models apart from the first model's *result
      parameters* are parameters of the 2D model.
    - All derived results from the two models are included in the :class Model2D:'s
      derived results.
    - By default, a parameter/derived result named `param` from a model named `model` is
      exposed as a parameter / result of the :class Model2D: named `param_model`. Custom
      naming schemes are possible by passing a `param_renames` dictionary into
      :meth __init__:.

    See also :class Fitter2D:.
    """

    def __init__(
        self,
        models: Tuple[Model, Model],
        result_params: Tuple[str],
        model_names: Optional[Tuple[str, str]] = None,
        param_renames: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        :param models: Tuple containing the two :class Model:s to be combined to make
            the 2D model. The model instances are considered "owned" by the 2D model
            (they are not copied). They should not be referenced externally.
        :param result_params: tuple of names of "result parameters" of the first model,
            whose value is found by evaluating the second model. The order of parameter
            names in this tuple must match the order of result channels for the second
            model.
        :param model_names: optional tuple of names for the two models. These are used
            when generating names for fit results and derived parameters. Empty strings
            are allowed so long as the two models do not share any parameter names. If
            this argument is `None` the model names default to `x0` and `x1`
            respectively.
        :param param_renames: optional dictionary mapping names of parameters/derived
            results in the :class Model2D: to new parameter names to rename them with.
            This provides a means for more flexible parameter name schemes than the
            default `{param_name}_{model_name}` format. Parameters may be renamed to
            `None` to exclude them from the 2D model.
        """
        self.models = models
        self.result_params = result_params
        self.model_names = model_names if model_names is not None else ("x0", "x1")
        self.param_renames = param_renames or {}

        self.dropped_params = [
            old_name
            for old_name, new_name in self.param_renames.items()
            if new_name is None
        ]
        self.param_renames = {
            old_name: new_name
            for old_name, new_name in self.param_renames.items()
            if new_name is not None
        }

        missing = set(self.result_params) - set(self.models[0].parameters.keys())
        if missing:
            raise ValueError(
                "Result parameters must be parameters of the first model. Unrecognised "
                f"result parameter names are: {missing}"
            )

        if len(self.result_params) != self.models[1].get_num_y_channels():
            raise ValueError(
                f"{len(self.result_params)} parameters passed when second model "
                f"requires {self.models[1].get_num_y_channels()}"
            )

        model_parameters = ({}, {})  # new_param_name: param_data
        for model_idx, model in enumerate(self.models):
            model_name = self.model_names[model_idx]
            model_parameters[model_idx].update(
                {
                    f"{param_name}_{model_name}": param_data
                    for param_name, param_data in self.models[
                        model_idx
                    ].parameters.items()
                }
            )

        for param_name in self.result_params:
            del model_parameters[0][f"{param_name}_{self.model_names[0]}"]

        duplicates = set(model_parameters[0].keys()).intersection(
            set(model_parameters[1].keys())
        )
        if duplicates:
            raise ValueError(
                f"Duplicate parameter names found between the two models: {duplicates}."
                " Do the models have different suffixes?"
            )

        self.parameters = {}
        self.parameters.update(model_parameters[0])
        self.parameters.update(model_parameters[1])

        # Apply parameter renames
        missing = set(self.param_renames.keys()) - set(self.parameters.keys())
        if missing:
            raise ValueError(
                "Parameter renames do not correspond to any parameter of the 2D model: "
                f"{missing}"
            )

        renamed_parameters = {
            self.param_renames[param_name]: param_data
            for param_name, param_data in self.parameters.items()
            if param_name in self.param_renames.keys()
        }

        self.parameters = {
            param_name: param_data
            for param_name, param_data in self.parameters.items()
            if param_name not in self.param_renames.keys()
        }

        duplicates = set(renamed_parameters.keys()).intersection(self.parameters.keys())
        if duplicates:
            raise ValueError(
                "Parameter renames duplicate existing model parameter names: "
                f"{duplicates}"
            )

        self.parameters.update(renamed_parameters)

        # cache these so we don't need to calculate them each time we evaluate func
        # model_param_name: new_param_name
        model_0_param_map = {}
        for model_param_name in self.models[0].parameters.keys():
            if model_param_name in self.result_params:
                continue

            new_param_name = f"{model_param_name}_{self.model_names[0]}"
            new_param_name = self.param_renames.get(new_param_name, new_param_name)

            model_0_param_map[model_param_name] = new_param_name

        model_1_param_map = {}
        for model_param_name in self.models[1].parameters.keys():
            new_param_name = f"{model_param_name}_{self.model_names[1]}"
            new_param_name = self.param_renames.get(new_param_name, new_param_name)
            model_1_param_map[model_param_name] = new_param_name

        self.model_param_maps = (model_0_param_map, model_1_param_map)

    def __call__(self, x: TX2D, **kwargs: float) -> TY2D:
        """Evaluates the model.

        :param x: tuple of `(x_axis_0, x_axis_1)`
        :param kwargs: keyword arguments specify values for parameters of the 2D model.

        All model parameters which are not `fixed_to` a value by default must be
        specified. Any parameters which are not specified default to their `fixed_to`
        values.
        """
        args = {
            param_name: param_data.fixed_to
            for param_name, param_data in self.parameters.items()
            if param_data.fixed_to is not None
        }
        args.update(kwargs)
        return self.func(x, args)

    def func(self, x: TX2D, param_values: Dict[str, float]) -> TY2D:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        To use the model as a function outside of a fit, :meth __call__: generally
        provides a more convenient interface.

        :param x: tuple of `(x_axis_0, x_axis_1)`
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        x_ax_0, x_ax_1 = [np.array(x_ax) for x_ax in x]
        x_shape = [len(x_ax) for x_ax in x]

        model_0_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.model_param_maps[0].items()
        }
        model_1_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.model_param_maps[1].items()
        }

        y = np.zeros((self.get_num_y_channels(), x_shape[0], x_shape[1]))

        y_1 = np.atleast_2d(self.models[1].func(x_ax_1, model_1_values))

        for x_idx in range(len(x_ax_1)):
            model_0_values.update(
                {
                    param_name: y_1[param_idx, x_idx]
                    for param_idx, param_name in enumerate(self.result_params)
                }
            )
            y_idx = np.atleast_2d(self.models[0].func(x_ax_0, model_0_values))
            y[:, :, x_idx] = y_idx

        return y

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model."""
        return self.models[0].get_num_y_channels()


class Fitter2D:
    """Hierarchical fits to :class Model2D:s.

    Fitting proceeds as follows:
      - We fit the complete y-axis dataset to a :class RepeatedModel: based on the first
        of the :class Model2D:'s models. This performs a joint fit to all points on the
        second x-axis dimension.
      - The "result parameters" from the first model are used to create a second y-axis
        dataset, which the :class Model2D:'s second model is fit to.

    See :class Model2D: for more details. For example usage, see
    `test\test_multi_fitter`.
    """

    x: TX2D
    y: TY2D
    sigma: TY2D
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    derived_values: Dict[str, float]
    derived_uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model2D
    free_parameters: List[str]

    def __init__(
        self,
        x: TX2D,
        y: TY2D,
        model: Model2D,
        fitter_cls: Optional[Fitter] = None,
        fitter_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        :param x: x-axis data in the for of a tuple `(x_axis_0, x_axis_1)`
        :param y: y-axis input data in the form of an array:
          shaped as `[y_channels, x_axis_0, x_axis_1]`
        :param fitter_cls: optional fitter class to use for the first fit (the second
          fit is always a normal fit). Defaults to :class NormalFitter:.
        :param fitter_args: optional dictionary of keyword arguments to pass into the
          fitter class
        """
        x = list(x)
        self.x = x = tuple([np.array(x_ax) for x_ax in x])
        self.y = y = np.array(y, ndmin=3)

        self.model = copy.deepcopy(model)
        self.fitter_cls = fitter_cls or NormalFitter
        self.fitter_args = dict(fitter_args or {})

        if len(self.x) != 2:
            raise ValueError("Fitter2D requires 2 x-axes")

        if len(self.x[0]) != self.y.shape[1] or len(self.x[1]) != self.y.shape[2]:
            raise ValueError("x-axis and y axis shapes must match")

        x_ax_0, x_ax_1 = self.x

        # fit along first x axis
        model_0 = self.model.models[0]
        common_params = [
            param_name
            for param_name in model_0.parameters
            if param_name not in self.model.result_params
        ]
        model_0 = RepeatedModel(
            model=self.model.models[0],
            common_params=common_params,
            num_repetitions=len(x_ax_1),
            aggregate_results=True,
        )

        y_0 = np.moveaxis(self.y, -1, 0)
        y_0 = np.reshape(y_0, (np.prod(y_0.shape[0:2]), y_0.shape[2]))

        if self.fitter_args.get("sigma") is not None:
            sigma_0 = self.fitter_args["sigma"]
            sigma_0 = np.array(sigma_0, ndmin=3)
            sigma_0 = np.moveaxis(sigma_0, -1, 0)
            sigma_0 = np.reshape(
                sigma_0, (np.prod(sigma_0.shape[0:2]), sigma_0.shape[2])
            )
            self.fitter_args["sigma"] = sigma_0

        fit_0 = self.fitter_cls(x=x_ax_0, y=y_0, model=model_0, **self.fitter_args)

        # aggregate results
        y_1 = np.zeros((len(self.model.result_params), len(x_ax_1)), dtype=np.float64)
        sigma_1 = np.zeros_like(y_1) if fit_0.sigma is not None else None

        for param_idx, param_name in enumerate(self.model.result_params):
            y_param = np.array(
                [fit_0.values[f"{param_name}_{idx}"] for idx in range(len(x_ax_1))]
            )
            sigma_param = np.array(
                [
                    fit_0.uncertainties[f"{param_name}_{idx}"]
                    for idx in range(len(x_ax_1))
                ]
            )

            y_1[param_idx, :] = y_param

            if sigma_1 is not None:
                sigma_1[param_idx, :] = sigma_param

        fit_1 = NormalFitter(
            x=x_ax_1,
            y=y_1,
            model=self.model.models[1],
            sigma=sigma_1,
        )

        # aggregate results
        self.values: Dict[str, float] = {}
        self.uncertainties: Dict[str, float] = {}
        self.derived_values: Dict[str, float] = {}
        self.derived_uncertainties: Dict[str, float] = {}
        self.initial_values: Dict[str, float] = {}

        renames = self.model.param_renames
        for result_dict_name in [
            "values",
            "uncertainties",
            "derived_values",
            "derived_uncertainties",
            "initial_values",
        ]:
            result_dict = getattr(self, result_dict_name)
            model_0_results = getattr(fit_0, result_dict_name)
            model_1_results = getattr(fit_1, result_dict_name)

            if result_dict_name in ["values", "uncertainties", "initial_values"]:
                model_0_keys = self.model.models[0].parameters.keys()
                model_0_keys = [
                    key for key in model_0_keys if key not in self.model.result_params
                ]
                model_1_keys = self.model.models[1].parameters.keys()
            else:
                model_0_keys = model_0_results.keys()
                model_1_keys = model_1_results.keys()

            fit_0_values = {
                f"{param_name}_{self.model.model_names[0]}": model_0_results[param_name]
                for param_name in model_0_keys
            }
            fit_1_values = {
                f"{param_name}_{self.model.model_names[1]}": model_1_results[param_name]
                for param_name in model_1_keys
            }

            result_dict.update(fit_0_values)
            result_dict.update(fit_1_values)

            for old_name in self.model.dropped_params:
                if old_name in result_dict.keys():
                    del result_dict[old_name]

            for old_name, new_name in renames.items():
                if old_name not in result_dict.keys():
                    continue
                result_dict[new_name] = result_dict.pop(old_name)

        free_parameters = [
            f"{param_name}_{self.model.model_names[0]}"
            for param_name in fit_0.free_parameters
            if param_name in self.model.models[0].parameters.keys()
        ]
        free_parameters += [
            f"{param_name}_{self.model.model_names[1]}"
            for param_name in fit_1.free_parameters
        ]

        free_parameters = [
            param_name
            for param_name in free_parameters
            if param_name not in self.model.dropped_params
        ]
        free_parameters = [
            renames.get(param_name, param_name) for param_name in free_parameters
        ]
        self.free_parameters = free_parameters

        self.sigma = fit_0.calc_sigma()
        if self.sigma is not None:
            y_shape = self.y.shape
            self.sigma = np.reshape(self.sigma, (y_shape[2], y_shape[0], y_shape[1]))
            self.sigma = np.moveaxis(self.sigma, 0, -1)

        self.fits = (fit_0, fit_1)  # TODO: annotate and check against fit!

    def evaluate(
        self,
        plot_mode=False,
        x_fit: Optional[TX2D] = None,
    ) -> Union[
        Tuple[TX2D, TY2D],
        Tuple[
            Array[("num_samples_ax_0",), np.float64],
            Array[("num_samples_ax_1",), np.float64],
            Array[("num_samples_ax_1", "num_samples_ax_0"), np.float64],
        ],
    ]:
        """Evaluates the model function using the fitted parameter set.

        :param unpack: if True, we format the output data to be convenient for plotting
            with matplotlib.pcolormesh by splatting the x-axis list and squeezing +
            transposing the y-axis data.
        :param x_fit: optional x-axis points to evaluate the model at. If
            `None`, we use the values stored as attribute `x` of the fitter.

        :returns: tuple of x-axis values used and corresponding y-axis values
            of the fitted model
        """
        x_fit = x_fit if x_fit is not None else self.x
        y_fit = self.model(x_fit, **self.values)

        if plot_mode:
            return *x_fit, y_fit.squeeze().T

        return x_fit, y_fit

    def residuals(self) -> TY2D:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]
