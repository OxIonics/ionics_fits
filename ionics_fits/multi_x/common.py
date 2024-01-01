import copy
import logging
from typing import Dict, Optional, Tuple, Union, Type, TypeVar, TYPE_CHECKING

import numpy as np


from .. import BinomialFitter, Fitter, Model, NormalFitter
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


logger = logging.getLogger(__name__)


class Model2D(Model):
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

    Notes:
    - All parameters from the two models apart from the first model's *result
      parameters* are parameters of the 2D model.
    - All derived results from the two models are included in the :class Model2D:'s
      derived results.
    - By default, a parameter/derived result named `param` from a model named `model` is
      exposed as a parameter / result in the :class Model2D: named `param_model`. Custom
      naming schemes are possible by passing a `param_renames` dictionary into
      :meth __init__:.
    """

    def __init__(
        self,
        models: Tuple[Model, Model],
        result_params: Tuple[str],
        model_names: Optional[Tuple[str, str]] = None,
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
            this argument is `None` the model names default to `x0` and `x1`
            respectively.
        :param param_renames: optional dictionary mapping names of parameters/derived
            results in the :class Model2D: to new parameter names to rename them with.
            This provides a means for more flexible parameter name schemes than the
            default `{param_name}_{model_name}` format.
        """
        self.__models = models
        self.__result_params = result_params
        self.__model_names = model_names if model_names is not None else ("x", "y")
        self.__param_renames = param_renames or {}

        self.__x_shape = None

        missing = set(self.__result_params) - set(self.__models[0].parameters.keys())
        if missing:
            raise ValueError(
                "Result parameters must be parameters of the first model. Unrecognised "
                f"result parameter names are: {missing}"
            )

        if len(self.__result_params) != self.__models[1].get_num_y_channels():
            raise ValueError(
                f"{len(self.__result_params)} parameters passed when second model "
                f"requires {self.__models[1].get_num_y_channels()}"
            )

        model_parameters = ({}, {})  # new_param_name: param_data
        for model_idx, model in enumerate(self.__models):
            model_name = self.__model_names[model_idx]
            model_parameters[model_idx].update(
                {
                    f"{param_name}_{model_name}": param_data
                    for param_name, param_data in self.__models[
                        model_idx
                    ].parameters.items()
                }
            )

        parameters = {}
        internal_parameters = [
            model_parameters[0].pop(f"{param_name}_{self.__model_names[0]}")
            for param_name in self.__result_params
        ]

        duplicates = set(model_parameters[0].keys()).intersection(
            set(model_parameters[1].keys())
        )
        if duplicates:
            raise ValueError(
                f"Duplicate parameter names found between the two models: {duplicates}."
                " Do the models have different suffixes?"
            )

        parameters.update(model_parameters[0])
        parameters.update(model_parameters[1])

        # Apply parameter renames
        missing = set(self.__param_renames.keys()) - set(parameters.keys())
        if missing:
            raise ValueError(
                "Parameter renames do not correspond to any parameter of the 2D model: "
                f"{missing}"
            )

        renamed_parameters = {
            self.__param_renames[param_name]: param_data
            for param_name, param_data in parameters.items()
            if param_name in self.__param_renames.keys()
        }

        parameters = {
            param_name: param_data
            for param_name, param_data in parameters.items()
            if param_name not in self.__param_renames.keys()
        }

        duplicates = set(renamed_parameters.keys()).intersection(parameters.keys())
        if duplicates:
            raise ValueError(
                "Parameter renames duplicate existing model parameter names: "
                f"{duplicates}"
            )

        parameters.update(renamed_parameters)

        # cache these so we don't need to calculate them each time we evaluate func
        # model_param_name: new_param_name
        self.__model_0_param_map = {}
        for model_param_name in self.__models[0].parameters.keys():
            if model_param_name in self.__result_params:
                continue

            new_param_name = f"{model_param_name}_{self.__model_names[0]}"
            new_param_name = self.__param_renames.get(new_param_name, new_param_name)

            self.__model_0_param_map[model_param_name] = new_param_name

        self.__model_1_param_map = {}
        for model_param_name in self.__models[1].parameters.keys():
            new_param_name = f"{model_param_name}_{self.__model_names[1]}"
            new_param_name = self.__param_renames.get(new_param_name, new_param_name)
            self.__model_1_param_map[model_param_name] = new_param_name

        self.__model_param_maps = (self.__model_0_param_map, self.__model_1_param_map)

        super().__init__(parameters=parameters, internal_parameters=internal_parameters)

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
        x_ax_0, x_ax_1 = x
        x_shape = [len(x_ax) for x_ax in x]

        model_0_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.__model_0_param_map.items()
        }
        model_1_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.__model_1_param_map.items()
        }

        y = np.zeros((x_shape[1], x_shape[0], self.get_num_y_channels()))

        y_1 = np.atleast_2d(self.__models[1].func(x_ax_1, model_1_values))

        for x_idx in range(len(x_ax_1)):
            model_0_values.update(
                {
                    param_name: y_1[param_idx, x_idx]
                    for param_idx, param_name in enumerate(self.__result_params)
                }
            )
            y_idx = np.atleast_2d(self.__models[0].func(x_ax_0, model_0_values))
            y[x_idx, :, :] = y_idx.T

        return y

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model."""
        return self.__models[0].get_num_y_channels()

    def get_num_x_axes(self) -> int:
        """Returns the number of x-axes supported by the model."""
        return 2

    def can_rescale(self) -> Tuple[bool, bool]:
        rescale_x, rescale_y = zip(*[model.can_rescale() for model in self.__models])
        return all(rescale_x), all(rescale_y)

    def estimate_parameters(self, x: TX2D, y: TY2D):
        x_ax_0 = x[0]
        x_ax_1 = x[1]
        x_shape = [len(x_ax) for x_ax in x]

        model_0 = self.__models[0]
        parameters = model_0.parameters

        heuristics = {
            param_name: np.zeros(len(x_ax_1 + 1)) for param_name in parameters.keys()
        }

        for x_1_idx in len(x_ax_0):
            model_0.parameters = copy.deepcopy(model_0.parameters)  # reset heuristics
            model_0.estimate_parameters(x=x_ax_0, y=y[x_1_idx, :, :])

            for param_name, param_data in model_0.parameters.items():
                heuristics[param_name][x_1_idx] = param_data.get_initial_value()

        for param_name in parameters.keys():
            heuristics[param_name][-1] = np.mean(heuristics[param_name][:-1])

        costs = np.zeros(len(x_ax_1) + 1)
        for idx in range(len(costs)):
            y_idx = self.__call__(
                x=x_ax_0,
                **{
                    param_name: heuristics[param_name][idx]
                    for param_name in self.heuristics.keys()
                },
            )
            costs[idx] = np.sqrt(np.sum((y - y_idx) ** 2))
        best_heuristic = np.argmin(costs)

        model_0.parameters = parameters
        for param_name, param_data in model_0.parameters.items():
            param_data.heuristic = heuristics[param_name][best_heuristic]

        model_1 = self.__models[1]
        y_1 = np.zeros(len(x_ax_1), model_1.get_num_y_channels())
        for idx, param_name in enumerate(self.__result_params):
            y_1[idx, :] = heuristics[param_name][:-1]

        model_1.estimate_parameters(x_ax_1, y_1)

    def calculate_derived_params(
        self,
        x: TX2D,
        y: TY2D,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_values = {}
        derived_uncertainties = {}

        for idx, model in enumerate(self.__models.values()):

            # model_param_name: new_param_name
            param_map = self.__model_param_maps[idx]

            model_values = {
                model_param_name: fitted_params[new_param_name]
                for model_param_name, new_param_name in param_map.items()
            }
            model_uncertainties = {
                model_param_name: fit_uncertainties[new_param_name]
                for model_param_name, new_param_name in param_map.items()
            }

            derived = model.calculate_derived_params(
                x=x[0],
                y=None,  # No good y-axis value to use for this
                fitted_params=model_values,
                fit_uncertainties=model_uncertainties,
            )
            model_derived_values, derived_uncertainties = derived

            derived_values.update(
                {
                    new_param_name: model_derived_values[model_param_name]
                    for model_param_name, new_param_name in param_map.items()
                }
            )
            derived_uncertainties.update(
                {
                    new_param_name: model_derived_values[model_param_name]
                    for model_param_name, new_param_name in param_map.items()
                }
            )

        derived_values = {
            self.__param_renames.get(param_name, param_name): value
            for param_name, value in derived_values.items()
        }
        derived_uncertainties = {
            self.__param_renames.get(param_name, param_name): value
            for param_name, value in derived_uncertainties.items()
        }

        return derived_values, derived_uncertainties


# TFitter = TypeVar("TFitter", bound=Type[Fitter])


# def make_2d_fitter(base_class: TFitter) -> TFitter:
#     """
#     Converts a :class Fitter: into one that is compatible with :class Model2D:s.
#     """
#     class Fitter2D(base_class):
#         def __init__(self, x: TX2D, y: TY2D, model: Model2D, **kwargs):
#             """
#             :param x: x-axis data in the for of a tuple `(x_axis_0, x_axis_1)`
#             :param y: y-axis input data in the form of an array:
#               shaped as `[x_axis_1, x_axis_0, y_channels]`
#             :param kwargs: passed to the fitter base class
#             """
#             y = np.atleast_3d(y)

#             if len(x) != 2:
#                 raise ValueError("Fitter2D requires 2 x-axes")

#             if len(x[0]) != y.shape[1] or len(x[1]) != y.shape[0]:
#                 raise ValueError("x-axis and y axis shapes must match")

#             x_shape = tuple([len(x_ax) for x_ax in x])
#             y_flat = np.reshape(y, (np.prod(x_shape), model.get_num_y_channels())).T

#             # Work around the fact that `Fitter` does not (currently) support working
#             # with multi-dimensional x-axis data.
#             # TODO: consider making the standard fitter work directly with
#             # multi-dimensional x-axis data

#             model = copy.deepcopy(model)

#             x_2d = x
#             model_func = model.func
#             model_estimate_parameters = model.estimate_parameters
#             calculate_derived_params = model.calculate_derived_params

#             def func_wrapped(self, x: TXFLAT, param_values: Dict[str, float]) -> TYFLAT:
#                 y = model_func(x=x2d, param_values=param_values)
#                 y = np.reshape(y, sum(x_shape), self.get_num_y_channels()).Tuple
#                 return y

#             def estimate_parameters(self, x: TX2D, y: TY2D):
#                 return model_estimate_parameters(x2d, )

#             super().__init__(
#                 x=np.arange((y.shape[1] * y.shape[0])),  # this does not scale properly!
#                 y=y_flat,
#                 model=model,
#                 **kwargs)

#             self.x = x
#             self.y = y

#     return Fitter2D


# NormalFitter2D = make_2d_fitter(NormalFitter)
# BinomialFitter2D = make_2d_fitter(BinomialFitter)
