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


class Model2D(Model):
    """Combines a pair of :class Model:s, which are each a function of 1 x-axis, to
    make a new :class Model:, which is a function of 2 x-axes.

    All y-axis data is generated from the output of the first model; the output from
    the second model provides the values of certain "result" parameters used by the
    first model. In other words:
      ```
      model_0 = models[0] = f(x_axis_0)
      model_1 = models[1] = g(x_axis_1)
      y(x_0, x_1) = model_0(x_0 | result_params = model_1(x_1))
      ```

    An intrinsic limitation of this approach is that the 2D fit function must be
    separable into functions of the two axes. This means, for example, that the 1D
    fit-functions must be aligned with the  two x axes. For example, one can't represent
    a non axis-aligned Gaussian in this way.

    Model parameters and results:
    - All parameters from the two models - apart from the first model's *result
      parameters* - are parameters of the 2D model.
    - All derived results from the two models are included in the :class Model2D:'s
      derived results.
    - A parameter/derived result named `param` from a model named `model` is
      exposed as a parameter / result of the :class Model2D: named `param_model`.
    - A :class WrappedModel: can be used to provide custom naming schemes
    """

    def __init__(
        self,
        models: Tuple[Model, Model],
        result_params: Tuple[str],
        model_names: Optional[Tuple[str, str]] = None,
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
            respectively. If a model name is an empty string, the trailing underscore
            is omitted in parameter / result names.
        """
        self.models = models
        self.result_params = result_params
        self.model_names = model_names if model_names is not None else ("x0", "x1")

        missing = set(self.result_params) - set(self.models[0].parameters.keys())
        if missing:
            raise ValueError(
                "Result parameters must be parameters of the first model. Unrecognised "
                f"result parameter names are: {missing}"
            )

        if len(self.result_params) != self.models[1].get_num_y_channels():
            raise ValueError(
                f"{len(self.result_params)} result parameters passed when second model "
                f"requires {self.models[1].get_num_y_channels()}"
            )

        # Map the parameters of the 1D models onto parameters of the 2D models
        parameters = {}
        internal_parameters = [
            self.models[0].parameters[param_name] for param_name in self.result_params
        ]

        self.model_param_maps = ({}, {})  # model_param_name: new_param_name

        for model_idx, model_name in enumerate(self.model_names):
            # Result parameters are internal parameters of the 2D model
            exclusions = self.result_params if model_idx == 0 else []
            suffix = f"_{model_name}" if model_name else ""
            self.model_param_maps[model_idx].update(
                {
                    model_param_name: f"{model_param_name}{suffix}"
                    for model_param_name in self.models[model_idx].parameters.keys()
                    if model_param_name not in exclusions
                }
            )

        duplicates = set(self.model_param_maps[0].values()).intersection(
            set(self.model_param_maps[1].values())
        )
        if duplicates:
            raise ValueError(
                f"Duplicate parameter names found between the two models: {duplicates}."
            )

        # Create the parameters / internal parameters for the 2D model
        for model_idx, param_map in enumerate(self.model_param_maps):
            model = self.models[model_idx]
            parameters.update(
                {
                    new_param_name: model.parameters[model_param_name]
                    for model_param_name, new_param_name in param_map.items()
                }
            )
            internal_parameters += model.internal_parameters

        super().__init__(parameters=parameters, internal_parameters=internal_parameters)

    def can_rescale(self) -> Tuple[bool, bool]:
        raise NotImplementedError

    def get_num_y_channels(self) -> int:
        return self.models[0].get_num_y_channels()

    def estimate_parameters(
        self,
        x: TX2D,
        y: TY2D,
    ):
        raise NotImplementedError

    def calculate_derived_params(
        self,
        x: TX2D,
        y: TY2D,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        raise NotImplementedError

    def func(self, x: TX2D, param_values: Dict[str, float]) -> TY2D:
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
        self.fits = (fit_0, fit_1)
        self.values: Dict[str, float] = {}
        self.uncertainties: Dict[str, float] = {}
        self.derived_values: Dict[str, float] = {}
        self.derived_uncertainties: Dict[str, float] = {}
        self.initial_values: Dict[str, float] = {}
        self.free_parameters: List[str] = []

        for result_dict_name in ["values", "uncertainties", "initial_values"]:
            result_dict = getattr(self, result_dict_name)
            for model_idx, model in enumerate(self.model.models):
                fit_results = getattr(self.fits[model_idx], result_dict_name)
                param_map = self.model.model_param_maps[model_idx]
                result_dict.update(
                    {
                        new_param_name: fit_results[model_param_name]
                        for model_param_name, new_param_name in param_map.items()
                    }
                )

        for model_idx, model in enumerate(self.model.models):
            param_map = self.model.model_param_maps[model_idx]
            model_name = self.model.model_names[model_idx]
            suffix = f"_{model_name}" if model_name else ""
            fit = self.fits[model_idx]

            self.derived_values.update(
                {
                    f"{model_param_name}{suffix}": value
                    for model_param_name, value in fit.derived_values.items()
                }
            )
            self.derived_uncertainties.update(
                {
                    f"{model_param_name}{suffix}": value
                    for model_param_name, value in fit.derived_uncertainties.items()
                }
            )

            self.free_parameters += [
                param_map[model_param_name]
                for model_param_name, new_param_name in param_map.items()
                if model_param_name in fit.free_parameters
            ]

        self.sigma = fit_0.calc_sigma()
        if self.sigma is not None:
            y_shape = self.y.shape
            self.sigma = np.reshape(self.sigma, (y_shape[2], y_shape[0], y_shape[1]))
            self.sigma = np.moveaxis(self.sigma, 0, -1)

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
