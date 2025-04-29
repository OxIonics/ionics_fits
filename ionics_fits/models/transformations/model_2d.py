from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...common import TX, TY, Model
from ...utils import TX_SCALE, TY_SCALE, to_float


class Model2D(Model):
    r"""Combines a pair of :class:`~ionics_fits.common.Model`\'s, each of which is a
    function of a single x-dimension, to make a new :class:`~ionics_fits.common.Model`\,
    which is a function of 2 x-axis dimensions.

    All y-axis data is generated from the output of the first model; the output from
    the second model provides the values of certain "result" parameters used by the
    first model. In other words::

      model_0 = models[0] = f(x_axis_0)
      model_1 = models[1] = g(x_axis_1)
      y(x_0, x_1) = model_0(x_0 | result_params = model_1(x_1))

    The 2D models generated using this class are separable into functions of the two
    x-axes. As a result, it can only generate axis-aligned models.

    This model provides a quick and convenient means of extending the x-axis
    dimensionality of existing models. The design aim is to maximise flexibility while
    making minimal assumptions about the underlying
    :class:`~ionics_fits.common.Model`\s.

    Model parameters and results:

    * All parameters from the two models - apart from the first model's *result
      parameters* - are parameters of the 2D model.
    * All derived results from the two models are included in the :class:`Model2D`\'s
      derived results.
    * A parameter/derived result named ``param`` from a model named ``model`` is
      exposed as a parameter / result of the :class:`Model2D` named ``param_model``.
    * A :class:`~ionics_fits.models.transformations.mapped_model.MappedModel` can be
      used to provide custom naming schemes

    Example usage:

    .. doctest::

        from ionics_fits.models.gaussian import Gaussian
        from ionics_Fits.models.transformations.model_2d import Model2D

        Gaussian2D = Model2D(
            models=(Gaussian(), Gaussian),
            result_params=("a_x0",),
        )

    This fits a Gaussian model along the first x-axis dimension. It then takes the
    amplitudes of those Gaussians at each value of the second x-axis dimension and
    fits those to a Gaussian to produce a 2D Gaussian fit.

    At present this class aggregates a pair of 1D models to make a 2D model. It would
    be easy to extend it to the more general case of aggregating an arbitrary number of
    models, each with arbitrary x-axis dimensionality. This has not been done yet
    because there has not been a use-case.

    See also :py:mod:`~ionics_fits.models.multi_x`.
    """

    def __init__(
        self,
        models: Tuple[Model, Model],
        result_params: Tuple[str],
        model_names: Optional[Tuple[str, str]] = None,
    ):
        r"""
        :param models: Tuple containing the two :class:`~ionics_fits.common.Model`\s to
            be combined to make the 2D model. The model instances are considered "owned"
            by the 2D model (they are not copied). They should not be referenced
            externally.
        :param result_params: tuple of names of "result parameters" of the first model,
            whose value is found by evaluating the second model. The order of parameter
            names in this tuple must match the order of y-axis dimensions for the second
            model.
        :param model_names: optional tuple of names for the two models. These are used
            when generating names for fit results and derived parameters. Empty strings
            are allowed so long as the two models do not share any parameter names. If
            this argument is ``None`` the model names default to ``x0`` and ``x1``
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

        if len(self.result_params) != self.models[1].get_num_y_axes():
            raise ValueError(
                f"{len(self.result_params)} result parameters passed when second model "
                f"requires {self.models[1].get_num_y_axes()}"
            )

        if any([model.get_num_x_axes() != 1 for model in self.models]):
            raise ValueError("Model2D currently only supports 1D models as inputs")

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
        self.wrap_scale_funcs()

    def wrap_scale_funcs(self):
        r"""Called during
        :meth:`~ionics_fits.models.transformations.model_2d.Model2D.__init__` to make
        sure that each :class:`~ionics_fits.common.ModelParameter` uses the scale factor
        from the appropriate x-axis when rescaling.

        This method wraps the :attr:`~ionics_fits.common.ModelParameter.scale_func`\s
        for each parameter of the 1D models to use the x-axis scale factor for their
        x-axis dimension when rescaling.

        This is appropriate where each model parameter scales with just that the x-axis
        associated with its model. If this is not the case, this method should be
        overridden. See :class:`~ionics_fits.models.multi_x.Gaussian2D` for an example
        of this.
        """

        def wrapped_scale_func(
            scale_func: Callable[[TX_SCALE, TY_SCALE], float],
            x_ax: int,
            x_scales: TX_SCALE,
            y_scales: TY_SCALE,
        ) -> float:
            return scale_func(np.array([x_scales[x_ax]]), y_scales)

        for model_idx, model in enumerate(self.models):
            parameters = list(model.parameters.values()) + model.internal_parameters
            for parameter in parameters:
                scale_func = parameter.scale_func
                parameter.scale_func = partial(
                    wrapped_scale_func, scale_func, model_idx
                )
                parameter.scale_func.__name__ = scale_func.__name__

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        rescale_x0, rescale_y0 = self.models[0].can_rescale()
        rescale_x1, rescale_y1 = self.models[1].can_rescale()

        rescale_x = list(np.hstack((rescale_x0, rescale_x1)))
        rescale_y = np.all(np.vstack((rescale_y0, rescale_y1)), axis=0)

        return rescale_x, rescale_y

    def get_num_x_axes(self) -> int:
        return 2

    def get_num_y_axes(self) -> int:
        return self.models[0].get_num_y_axes()

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        x = np.atleast_2d(x)

        model_0_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.model_param_maps[0].items()
        }
        model_1_values = {
            model_param_name: param_values[new_param_name]
            for model_param_name, new_param_name in self.model_param_maps[1].items()
        }

        x_1_ax = np.unique(x[1, :])
        result_param_values = self.models[1].func(x_1_ax, model_1_values)
        result_param_values = np.atleast_2d(result_param_values)

        y = np.zeros((self.get_num_y_axes(), x.shape[1]), dtype=np.float64)
        for x_1_idx, x_1 in np.ndenumerate(x_1_ax):
            x_0_mask = x[1, :] == x_1
            x_0_ax = x[0, x_0_mask]
            model_0_values.update(
                {
                    param_name: to_float(result_param_values[param_idx, x_1_idx])
                    for param_idx, param_name in enumerate(self.result_params)
                }
            )
            y[:, x_0_mask] = np.atleast_2d(self.models[0].func(x_0_ax, model_0_values))

        return y

    def estimate_parameters(self, x: TX, y: TY):
        x_1_ax = np.unique(x[1, :])

        # Step 1: find heuristics for model 0 for each point along the second x-axis
        x_1_keep = np.full(len(x_1_ax) + 1, True, dtype=bool)
        heuristics = {
            param_name: np.zeros(len(x_1_ax) + 1, dtype=np.float64)
            for param_name in self.models[0].parameters.keys()
        }

        for x_1_idx, x_1 in np.ndenumerate(x_1_ax):
            x_1_mask = x[1, :] == x_1
            x_0_ax = x[0, x_1_mask]
            y_x_1 = y[:, x_1_mask]

            self.models[0].clear_heuristics()
            try:
                self.models[0].estimate_parameters(x=x_0_ax, y=y_x_1)
            except Exception:
                x_1_keep[x_1_idx] = False

            for param_name, param_data in self.models[0].parameters.items():
                heuristics[param_name][x_1_idx] = param_data.get_initial_value()

        x_1_ax = x_1_ax[x_1_keep[:-1]]
        for param_name in heuristics.keys():
            heuristics[param_name] = heuristics[param_name][x_1_keep]

        for param_name in heuristics.keys():
            heuristics[param_name][-1] = np.mean(heuristics[param_name][:-1])

        # Step 2: pick the best heuristic
        costs = np.zeros(len(x_1_ax) + 1, dtype=np.float64)
        for heuristic_idx in range(len(costs)):
            y_heuristic = np.zeros_like(y)
            x_1_heuristics = {
                param_name: param_heuristic[heuristic_idx]
                for param_name, param_heuristic in heuristics.items()
            }

            for x_1_idx, x_1 in np.ndenumerate(x_1_ax):
                x_1_heuristics.update(
                    {
                        param_name: heuristics[param_name][x_1_idx]
                        for param_name in self.result_params
                    }
                )

                x_1_mask = x[1, :] == x_1
                x_0_ax = x[0, x_1_mask]
                y_heuristic[:, x_1_mask] = self.models[0].func(
                    x=x_0_ax, param_values=x_1_heuristics
                )
            costs[heuristic_idx] = np.sqrt(np.sum((y - y_heuristic) ** 2))

        lowest_cost = np.argmin(costs)
        best_heuristic = {
            param_name: heuristics[param_name][lowest_cost]
            for param_name in heuristics.keys()
        }

        for model_param_name, new_param_name in self.model_param_maps[0].items():
            param_heuristic = best_heuristic[model_param_name]
            self.parameters[new_param_name].heuristic = param_heuristic

        # Step 3: use the result parameter heuristics from model 0 as an input to
        # model 1
        y_1 = np.zeros((len(self.result_params), len(x_1_ax)), dtype=np.float64)
        for param_idx, result_param_name in enumerate(self.result_params):
            y_1[param_idx, :] = heuristics[result_param_name][:-1]

        self.models[1].estimate_parameters(x_1_ax, y_1)

        for model_param_name, new_param_name in self.model_param_maps[1].items():
            param_data = self.models[1].parameters[model_param_name]
            heuristic = param_data.get_initial_value()
            self.parameters[new_param_name].heuristic = heuristic

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_values = {}
        derived_uncertainties = {}

        model_values = ({}, {})
        model_uncertainties = ({}, {})
        model_derived_values = ({}, {})
        model_derived_uncertainties = ({}, {})

        # Result parameters do not have meaningful fitted values / uncertainties
        for param_name in self.result_params:
            model_values[0][param_name] = float("nan")
            model_uncertainties[0][param_name] = float("nan")

        for model_idx, model in enumerate(self.models):
            param_map = self.model_param_maps[model_idx]
            model_values[model_idx].update(
                {
                    model_param_name: fitted_params[new_param_name]
                    for model_param_name, new_param_name in param_map.items()
                }
            )
            model_uncertainties[model_idx].update(
                {
                    model_param_name: fit_uncertainties[new_param_name]
                    for model_param_name, new_param_name in param_map.items()
                }
            )

            derived = model.calculate_derived_params(
                x=x[model_idx, :],
                y=y,
                fitted_params=model_values[model_idx],
                fit_uncertainties=model_uncertainties[model_idx],
            )

            if self.model_names[model_idx] != "":
                model_suffix = f"_{self.model_names[model_idx]}"
            else:
                model_suffix = ""

            model_derived_values[model_idx].update(
                {
                    f"{param_name}{model_suffix}": value
                    for param_name, value in derived[0].items()
                }
            )
            model_derived_uncertainties[model_idx].update(
                {
                    f"{param_name}{model_suffix}": value
                    for param_name, value in derived[1].items()
                }
            )

        duplicates = set(model_derived_values[0].keys()).intersection(
            set(model_derived_values[1].keys())
        )
        if duplicates:
            raise ValueError(
                "Duplicate derived result names found between the two models: "
                f"{duplicates}."
            )

        derived_values = {}
        derived_values.update(model_derived_values[0])
        derived_values.update(model_derived_values[1])

        derived_uncertainties = {}
        derived_uncertainties.update(model_derived_uncertainties[0])
        derived_uncertainties.update(model_derived_uncertainties[1])

        return derived_values, derived_uncertainties
