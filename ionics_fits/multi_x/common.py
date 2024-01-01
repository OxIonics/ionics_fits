import copy
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np


from .. import Fitter, Model, NormalFitter
from ..utils import Array, ArrayLike
from ..models import RepeatedModel


if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_samples_x_flat = float
    num_samples_ax_0 = float
    num_samples_ax_1 = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


logger = logging.getLogger(__name__)


TX2D = Tuple[
    ArrayLike[("num_samples_ax_0",), np.float64],
    ArrayLike[("num_samples_ax_1",), np.float64],
]
TY2D = ArrayLike[("num_samples_ax_1", "num_samples_ax_0", "num_y_channels"), np.float64]

TXFLAT = ArrayLike[("num_samples_x_flat",), np.float64]
TYFLAT = ArrayLike[("num_y_channels", "num_samples_x_flat"), np.float64]

class Model2D:
    """ Base class providing a :class Model:-like interface for models with
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

    See also :class Fitter2D:.
    """

    def __init__(
        self,
        models: Tuple[Model, Model],
        result_params: Tuple[str],
        model_names: Optional[Tuple[str, str]] = None,
        common_params: Optional[Tuple[str]] = None,
        param_renames: Optional[Dict[str, str]] = None,
    ):
        """
        :param models: Tuple containing the two :class Model:s to be combined to make
            the 2D model.
        :param result_params: tuple of names of "result parameters" of the first model,
            whose value is found by evaluating the second model. The order of parameter
            names in this tuple must match the order of result channels for the second
            model.
        :param model_names: optional tuple of names for the two models. These are used
            when generating names for fit results and derived parameters. Empty strings
            are allowed so long as the two models do not share any parameter names. If
            this argument is `None` the model names default to `x` and `y` respectively.
        :param common_params: optional tuple of "common parameters" for the first model.
            These parameters are fit jointly (their value is the same for all points on
            `x_axis_1`). Any parameters of the first model not included here are treated
            as "independent" and may vary as a function of `x_axis_1`. See :class
            RepeatedModel: for further details.
        :param param_renames: optional dictionary mapping names of parameters in the 2D
            model to new names to rename them to. This allows more intuitive parameter
            names than the default `{param_name}_{model_name}` scheme.
        Notes:
        - All parameters from the two models apart from the first model's *result
          parameters* are parameters of the 2D model
        - A parameter named `param` from a model named `model` is exposed as a parameter
          of the 2D model named `param_model`
        - All derived results from the second model are exposed in the 2D model's
          derive parameter dictionary in the usual format (`{param_name}_{model_name}`)
        - Derived results from the first model are aggregated (see :class
          RepeatedModel`). Results whose value is the same for all points on `x_axis_1`
          are exposed in the usual format. Results whose value changes as a function of
          `x_axis_1` are omitted from the exposed dictionary.
        """
        self.models = models
        self.result_params = result_params
        self.model_names = model_names if model_names is not None else ("x", "y")
        self.common_params = common_params or ()
        self.param_renames = param_renames or {}

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

        for param_name in result_params:
            del model_parameters[0][f"{param_name}_{self.model_names[0]}"]

        duplicates = set(model_parameters[0].keys()).intersection(
            set(model_parameters[1].keys())
        )
        if duplicates:
            raise ValueError(
                f"Duplicate parameter names found between the two models: {duplicates}."
                " Do the models have different suffixes?"
            )

        parameters = model_parameters[0]
        parameters.update(model_parameters[1])

        missing = set(self.param_renames.keys()) - set(parameters.keys())
        if missing:
            raise ValueError(
                "Parameter renames do not correspond to any parameter of the 2D model: "
                f"{missing}"
            )

        self.parameters = {
            param_name: param_data
            for param_name, param_data in parameters.items()
            if param_name not in self.param_renames.keys()
        }

        duplicates = set(self.param_renames.keys()).intersection(self.parameters.keys())
        if duplicates:
            raise ValueError(
                "Parameter renames duplicate existing model parameter names: "
                f"{duplicates}"
            )

        self.parameters.update(
            {
                new_param_name: parameters[old_param_name]
                for old_param_name, new_param_name in self.param_renames.items()
            }
        )

        # cache these so we don't need to calculate them each time we evaluate func
        # model_param_name: new_param_name
        self.__model_0_param_map = {}
        for model_param_name in self.models[0].parameters.keys():
            if model_param_name in self.result_params:
                continue

            new_param_name = f"{model_param_name}_{self.model_names[0]}"
            new_param_name = self.param_renames.get(new_param_name, new_param_name)

            self.__model_0_param_map[model_param_name] = new_param_name

        self.__model_1_param_map = {}
        for model_param_name in self.models[1].parameters.keys():
            new_param_name = f"{model_param_name}_{self.model_names[1]}"
            new_param_name = self.param_renames.get(new_param_name, new_param_name)

            self.__model_1_param_map[model_param_name] = new_param_name

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

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model."""
        return self.models[0].get_num_y_channels()

    def func(self, x: TX2D, param_values: Dict[str, float]) -> TY2D:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        To use the model as a function outside of a fit, :meth __call__: generally
        provides a more convenient interface.

        :param x: tuple of `(x_axis_0, x_axis_1)`
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        model_0_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.__model_0_param_map.items()
        }
        model_1_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.__model_1_param_map.items()
        }

        x_ax_0, x_ax_1 = x
        y = np.zeros((len(x[1]), len(x[0]), self.get_num_y_channels()))

        y_1 = np.atleast_2d(self.models[1].func(x_ax_1, model_1_values))

        for x_idx in range(len(x_ax_1)):
            model_0_values.update(
                {
                    param_name: y_1[param_idx, x_idx]
                    for param_idx, param_name in enumerate(self.result_params)
                }
            )
            y_idx = np.atleast_2d(self.models[0].func(x_ax_0, model_0_values))
            y[x_idx, :, :] = y_idx.T

        return y


class Fitter2D(Fitter):
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
    fits: Tuple[Fitter, NormalFitter]
    sigma: Array[("num_y_channels", "num_samples"), np.float64]
    values: Dict[str, float]
    uncertainties: Dict[str, float]
    derived_values: Dict[str, float]
    derived_uncertainties: Dict[str, float]
    initial_values: Dict[str, float]
    model: Model
    free_parameters: List[str]

    def __init__(
        self,
        x: TX2D,
        y: TY2D,
        model: Model2D,
        fitter_cls: Optional[Fitter] = None,
        fitter_args: Optional[Dict] = None,
    ):
        """
        :param x: tuple of `(x_axis_0, x_axis_1)`
        :param y: y-axis input data
        :param fitter_cls: optional fitter class to use for the first fit (the second
          fit is always a normal fit). Defaults to :class NormalFitter:.
        :param fitter_args: optional dictionary of keyword arguments to pass into the
          fitter class
        """
        self.fitter_cls = fitter_cls or NormalFitter
        self.fitter_args = dict(fitter_args or {})
        self.model = copy.deepcopy(model)
        self.x = x

        y = np.atleast_3d(y)

        if len(x) != 2:
            raise ValueError("Fitter2D requires 2 x-axes")

        if len(x[0]) != y.shape[1] or len(x[1]) != y.shape[0]:
            raise ValueError("x-axis and y axis shapes must match")

        x_ax_0 = x[0]
        x_ax_1 = x[1]

        # fit along first axis
        model_0 = RepeatedModel(
            model=self.model.models[0],
            common_params=self.model.common_params,
            num_repetitions=len(x_ax_1),
            aggregate_results=True,
        )

        y_0 = np.moveaxis(y, -1, 0)
        y_0 = np.reshape(y_0, (np.prod(y_0.shape[0:2]), y_0.shape[2]))
        fit_0 = self.fitter_cls(x=x_ax_0, y=y_0, model=model_0, **self.fitter_args)

        # aggregate results
        y_1 = np.zeros((len(self.model.result_params), len(x_ax_1)), dtype=np.float64)
        sigma_1 = np.zeros_like(y_1)
        for param_idx, param_name in enumerate(self.model.result_params):
            y_param = np.array(
                [fit_0.values[f"{param_name}_{idx}"] for idx in range(len(x_ax_1))]
            )
            sigma_param = np.array(
                [fit_0.values[f"{param_name}_{idx}"] for idx in range(len(x_ax_1))]
            )

            y_1[param_idx, :] = y_param
            sigma_1[param_idx, :] = sigma_param

        y_1 = y_1
        sigma_1 = sigma_1

        fit_1 = NormalFitter(
            x=x_ax_1,
            y=y_1,
            model=self.model.models[1],
            sigma=sigma_1,
            **self.fitter_args,
        )

        def aggregate_results(
            attr: str, fit_0_param_filter: List[str], renames: Dict[str, str], aggregated_renames
        ):
            fit_0_results = {
                    f"{param_name}_{self.model.model_names[0]}": value
                    for param_name, value in getattr(fit_0, attr).items()
                }
            fit_1_results = {
                    f"{param_name}_{self.model.model_names[1]}": value
                    for param_name, value in getattr(fit_1, attr).items()
                }


            if fit_0_param_filter is not None:
                fit_0_param_filter = [aggregated_renames.get(param_name, param_name) for param_name in fit_0_param_filter]
                fit_0_results = {param_name: value for param_name, value in fit_0_results.items() if param_name in fit_0_param_filter}
            
            aggregated_results = fit_0_results
            aggregated_results.update(fit_1_results)
            if renames is not None:
                aggregated_results = {renames.get(param_name, param_name): value for param_name, value in aggregated_results.items()}

            setattr(self, attr, aggregated_results)

        # Aggregate fixed parameters of the first model

        # Model2D_param_name: model_0_param_name
        aggregated_renames = {
                param_name: f"{param_name}_0_{self.model.model_names[0]}"
                for param_name, param_data in self.model.models[0].parameters.items()
                if param_data.fixed_to is not None
            }

        for attr in ["values", "uncertainties", "initial_values"]:
            aggregate_results(
                attr,
                fit_0_param_filter=self.model.models[0].parameters.keys(),
                aggregated_renames=aggregated_renames,
                renames=self.model.param_renames,
            )
        for attr in [
            "derived_values",
            "derived_uncertainties",
        ]:
            aggregate_results(attr, fit_0_param_filter=None, renames=None, aggregated_renames=None)

        # self.models = [model_0, models[1]]

        # subtract independent variable?
        self.free_parameters = [
            param_name
            for param_name, param_data in self.model.models[0].parameters.items()
            if param_data.fixed_to is None
        ]
        # self.free_parameters += fit_1.free_parameters
        # self.initial_values =
        self.fits = (fit_0, fit_1)
        self.sigma = self.calc_sigma()
        # assert self.sigma.shape == self.y.shape

    def evaluate(
        self,
        transpose_and_squeeze=False,
        x_fit: Optional[TX2D] = None,
    ) -> Tuple[TX2D, TY2D]:
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

    def residuals(self) -> TY2D:
        """Returns an array of fit residuals."""
        return self.y - self.evaluate()[1]

    def calc_sigma(self) -> TY2D:
        """Return an array of standard error values for each y-axis data point."""
        return self.fits[0].calc_sigma()
