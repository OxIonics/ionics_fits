from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import param_like
from ...common import Model, ModelParameter, TX, TY


class AggregateModel(Model):
    """Model formed by aggregating one or more models.

    When aggregating a number of identical models, use a :class RepeatedModel: instead.

    Aggregate models have a number of uses. For example

    * joint fits to multiple data sets (pass the datasets in as y axes and use
      "common" parameters for any parameters which are fit jointly to all y axis
      dimensions).
    * fit multiple datasets simultaneously. This is useful, for example in automated
      tooling such as `ndscan`'s `OnlineAnalysis`.
    """

    def __init__(
        self,
        models: Dict[str, Model],
        common_params: Optional[
            Dict[str, Tuple[ModelParameter, List[Tuple[str, str]]]]
        ] = None,
    ):
        """
        :param models: The models to be aggregated. This should be a dictionary mapping
          model names to model instances. The model names are used as suffixes for names
          of model parameters and derived results. For example, if one of the aggregated
          models named `model` has a parameter `param`, the aggregate model will have a
          parameter `param_model`.

          The passed-in models are considered "owned" by the AggregateModel and should
          not be used / modified elsewhere.

        :param common_params: Optional dictionary specifying "common" model parameters.
          This feature allows multiple parameters (which can be from the same or
          different models) to be fit jointly to a single value. The common parameters
          are demoted to being `internal` model parameters and a new parameter is
          introduced to expose the common value to the user.

          The parameter metadata (limits, fixed_to, user_estimate, etc.) from the new
          parameter replaces the metadata for all parameters bound to it. Metadata set
          on the bound parameters is disregarded.

          The dictionary keys are the names of the new model parameters.

          The dictionary values are a tuple containing the new model template parameter
          and a list of parameters to bind to the new parameter. The bound parameter
          lists should be lists of tuples, specifying the parameters to bind to the new
          parameter. The tuples should contain two strings, specifying the name of the
          model which owns the common parameter, and the name of the model parameter to
          make common.

          The new model parameters inherit their metadata (limits etc.) from the
          template parameters, which are (deep) copied and are not modified.

        At present this class only supports models with a single y-axis dimension. This
        is just because no one got around to implementing it yet rather than any
        fundamental difficulty.
        """
        self.models = models

        axes = [model.get_num_x_axes() for model in self.models.values()]
        self.num_x_axes = axes[0]
        if axes.count(self.num_x_axes) != len(axes):
            raise ValueError("All models must have the same number of x-axes")

        # make model scale functions use correct y-axis dimension
        def wrapped_scale_func(model_idx, scale_func, x_scales, y_scales):
            y_scales_model = [y_scales[model_idx]]
            return scale_func(x_scales, y_scales_model)

        for idx, model in enumerate(self.models.values()):
            model_params = list(model.parameters.values()) + model.internal_parameters
            for parameter in model_params:
                wrapped_param_scale_func = partial(
                    wrapped_scale_func, idx, parameter.scale_func
                )
                parameter.scale_func = wrapped_param_scale_func

        common_params = common_params or {}

        # aggregate internal parameters from all models
        internal_parameters = []
        for model in self.models.values():
            internal_parameters += model.internal_parameters

        # organise the common parameter mapping data in ways that will be useful later

        # {new_param_name: [(model_name, param_name)]}
        self.common_param_list = {
            new_param_name: [] for new_param_name in common_params.keys()
        }

        # {model_name: [model_common_params]}
        self.model_common_params: Dict[str, List[str]] = {
            model_name: [] for model_name in self.models.keys()
        }

        # {(model_name, param_name): new_param_name}
        self.common_param_map: Dict[Tuple[str, str], str] = {}

        # {new_param_name: new_param}
        new_parameters: Dict[str, ModelParameter] = {}

        for new_param_name, (template_param, bound_params) in common_params.items():
            new_parameters[new_param_name] = param_like(template_param)
            for bind in bound_params:
                bound_model_name, bound_param_name = bind

                if bound_model_name not in self.models.keys():
                    raise ValueError(
                        f"Bound model name {bound_model_name} does not match any "
                        "aggregated model"
                    )

                model_parameters = self.models[bound_model_name].parameters
                if bound_param_name not in model_parameters.keys():
                    raise ValueError(
                        f"Bound parameter name {bound_param_name} does not match any "
                        "aggregated model"
                    )

                self.common_param_list[new_param_name] = [bind]
                self.model_common_params[bound_model_name].append(bound_param_name)
                self.common_param_map[bind] = new_param_name

        # aggregate non-common parameters from all models
        parameters: Dict[str, ModelParameter] = {}
        for model_name, model in self.models.items():
            if model.get_num_y_axes() != 1:
                raise ValueError(
                    "AggregateModel currently only supports models with a single y-axis"
                    " dimension."
                )
            parameters.update(
                {
                    f"{param_name}_{model_name}": param_data
                    for param_name, param_data in model.parameters.items()
                    if param_name not in self.model_common_params[model_name]
                }
            )

        duplicates = set(new_parameters.keys()).intersection(parameters.keys())
        if duplicates:
            raise ValueError(
                "New parameter names duplicate names of existing model parameters: "
                f"{duplicates}"
            )

        parameters.update(new_parameters)

        super().__init__(parameters=parameters, internal_parameters=internal_parameters)

    def get_num_x_axes(self) -> int:
        return self.num_x_axes

    def get_num_y_axes(self) -> int:
        return len(self.models)

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        rescale_xs, rescale_ys = zip(
            *[model.can_rescale() for model in self.models.values()]
        )
        rescale_xs = np.array(rescale_xs)
        rescale_ys = np.array(rescale_ys)

        rescale_x = list(np.all(rescale_xs, axis=0))
        rescale_ys = list(np.squeeze(rescale_ys))

        return rescale_x, rescale_ys

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        x = np.atleast_2d(x)
        num_samples = x.shape[1]
        ys = np.zeros((self.get_num_y_axes(), num_samples), dtype=np.float64)
        for idx, (model_name, model) in enumerate(self.models.items()):
            model_common_params = self.model_common_params[model_name]
            model_params = {
                param_name: param_values[f"{param_name}_{model_name}"]
                for param_name in model.parameters.keys()
                if param_name not in model_common_params
            }
            model_params.update(
                {
                    bound_param_name: param_values[
                        self.common_param_map[(model_name, bound_param_name)]
                    ]
                    for bound_param_name in model_common_params
                }
            )

            ys[idx, :] = np.atleast_2d(model.func(x, model_params))

        return ys

    def estimate_parameters(self, x: TX, y: TY):
        for idx, (model_name, model) in enumerate(self.models.items()):
            # replace bound model parameters with new ones based on our template
            # NB we don't do this in __init__ because we want to capture subsequent
            # changes to parameter metadata
            for bound_param_name in self.model_common_params[model_name]:
                new_param_name = self.common_param_map[(model_name, bound_param_name)]
                new_param = self.parameters[new_param_name]
                model.parameters[bound_param_name] = param_like(new_param)

            model.estimate_parameters(x, y[idx])

        # use the mean value from all models as our heuristic for common params
        for new_param_name, binds in self.common_param_list.items():
            estimates = [
                self.models[model_name].parameters[param_name].get_initial_value()
                for model_name, param_name in binds
            ]
            self.parameters[new_param_name].heuristic = np.mean(estimates)

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        for idx, (model_name, model) in enumerate(self.models.items()):
            model_common_params = self.model_common_params[model_name]
            model_fitted_params = {
                param_name: fitted_params[f"{param_name}_{model_name}"]
                for param_name in model.parameters.keys()
                if param_name not in model_common_params
            }
            model_fit_uncertainties = {
                param_name: fit_uncertainties[f"{param_name}_{model_name}"]
                for param_name in model.parameters.keys()
                if param_name not in model_common_params
            }

            model_fitted_params.update(
                {
                    bound_param_name: fitted_params[
                        self.common_param_map[(model_name, bound_param_name)]
                    ]
                    for bound_param_name in model_common_params
                }
            )
            model_fit_uncertainties.update(
                {
                    bound_param_name: fit_uncertainties[
                        self.common_param_map[(model_name, bound_param_name)]
                    ]
                    for bound_param_name in model_common_params
                }
            )

            derived = model.calculate_derived_params(
                x=x,
                y=y[idx, :],
                fitted_params=model_fitted_params,
                fit_uncertainties=model_fit_uncertainties,
            )
            model_derived_params, model_derived_uncertainties = derived
            derived_params.update(
                {
                    f"{param_name}_{model_name}": value
                    for param_name, value in model_derived_params.items()
                }
            )
            derived_uncertainties.update(
                {
                    f"{param_name}_{model_name}": value
                    for param_name, value in model_derived_uncertainties.items()
                }
            )

        return derived_params, derived_uncertainties
