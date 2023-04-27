from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class AggregateModel(Model):
    """Model formed by aggregating one or more models"""

    def __init__(
        self,
        models: List[Tuple[str, Model]],
        common_params: Optional[List[str]] = None,
    ):
        """
        :param models: The models to be aggregated. This should be a list of tuples,
          with each element containing a model name string and a model instance. The
          model names are used as prefixes for names of model parameters and derived
          results. For example, if one of the aggregated models named `foo` has a
          parameter `bar`, the aggregate model will have a parameter `foo_bar`. The
          aggregated model instances are considered to be "owned" by the aggregate model
          and may be mutated by it - for example to set heuristic values.

        At present this class only supports models with a single y channel. This is just
        because no one got around to implementing it yet rather than any fundamental
        difficulty.

        A possible future extension of this class would be to introduce a notion of
        "common parameters", whose value is the same for all of the aggregated models. A
        design question here is where the common parameters should get their properties
        (bounds, scale factors, etc) from and how we should deal with their heuristics.
        """
        self.models = models

        parameters = {}
        for model_name, model in self.models:
            if model.get_num_y_channels() != 1:
                raise ValueError(
                    "AggregateModel only supports models with a single y channel"
                )
            parameters.update(
                {
                    f"{model_name}_{param_name}": param
                    for param_name, param in model.parameters.items()
                }
            )

        super().__init__(parameters=parameters)

    def get_num_y_channels(self) -> int:
        return len(self.models)

    def func(
        self,
        x: Array[("num_samples",), np.float64],
        param_values: Dict[str, float],
    ) -> Array[("num_samples", "num_y_channels"), np.float64]:
        ys = []
        for model_name, model in self.models:
            model_params = {
                param_name: param_values[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }
            ys.append(model.func(x, model_params))

        return np.stack(ys).T

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        for idx, (model_name, model) in enumerate(self.models):
            params = {
                param_name: model_parameters[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }
            model.estimate_parameters(x, y[:, idx], params)

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        for idx, (model_name, model) in enumerate(self.models):
            model_fitted_params = {
                param_name: fitted_params[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }
            model_fitted_uncertainties = {
                param_name: fit_uncertainties[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }

            derived = model.calculate_derived_params(
                x=x,
                y=y[:, idx],
                fitted_params=model_fitted_params,
                fit_uncertainties=model_fitted_uncertainties,
            )
            model_derived_params, model_derived_uncertainties = derived

            derived_params.update(
                {
                    f"{model_name}_{param_name}": value
                    for param_name, value in model_derived_params.items()
                }
            )
            derived_uncertainties.update(
                {
                    f"{model_name}_{param_name}": value
                    for param_name, value in model_derived_uncertainties.items()
                }
            )

        return derived_params, derived_uncertainties
