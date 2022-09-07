from typing import Dict, TYPE_CHECKING

import numpy as np

from ..common import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float


class MappedModel(Model):
    """`Model` wrapping another `Model` with renamed parameters"""

    def __init__(
        self,
        inner: Model,
        mapped_params: Dict[str, str],
        fixed_params: Dict[str, float] = None,
    ):
        """Init

        :param inner: The wrapped model, the implementation of `inner` will be used
            after the parameter mapping has been done.
        :param mapped_params: dictionary mapping names of parameters in the new
            model to names of parameters used in the wrapped model.
        :param fixed_params: dictionary mapping names of parameters used in the
            wrapped model to values they are fixed to in the new model. These
            will not be parameters of the new model.
        """
        inner_params = inner.parameters

        if unknown_mapped_params := set(mapped_params.values()) - inner_params.keys():
            raise ValueError(
                "The target of parameter mappings must be parameters of the inner "
                f"model. The mapping targets are not: {unknown_mapped_params}"
            )

        if unknown_fixed_params := fixed_params.keys() - inner_params.keys():
            raise ValueError(
                "Fixed parameters must be parameters of the inner model. The "
                f"follow fixed parameters are not: {unknown_fixed_params}"
            )

        if missing_params := inner_params.keys() - (
            fixed_params.keys() | mapped_params.values()
        ):
            raise ValueError(
                "All parameters of the inner model must be either mapped of "
                "fixed. The following inner model parameters are neither: "
                f"{missing_params}"
            )

        if duplicated_params := fixed_params.keys() & mapped_params.values():
            raise ValueError(
                "Parameters cannot be both mapped and fixed. The following "
                f"parameters are both: {duplicated_params}"
            )

        params = {
            new_name: inner_params[old_name]
            for new_name, old_name in mapped_params.items()
        }
        super().__init__(parameters=params)
        self.inner = inner
        self.mapped_args = mapped_params
        self.fixed_params = fixed_params or {}

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        """Evaluates the model at a given set of x-axis points and with a given set of
        parameter values and returns the result.

        Overload this to provide a model function with a dynamic set of parameters,
        otherwise prefer to override `_func`.

        :param x: x-axis data
        :param param_values: dictionary of parameter values
        :returns: array of model values
        """
        new_params = {
            old_name: param_values[new_name]
            for new_name, old_name in self.mapped_args.items()
        }
        new_params.update(self.fixed_params)
        return self.inner.func(x, new_params)

    def _inner_estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        inner_parameters: Dict[str, ModelParameter],
    ) -> Dict[str, float]:
        return self.inner.estimate_parameters(x, y, inner_parameters)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ) -> Dict[str, float]:
        """
        Returns a dictionary of estimates for the model parameter values for the
        specified dataset. Typically called during `Fitter.fit`.

        The dataset must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values.

        :param x: x-axis data
        :param y: y-axis data
        :param model_parameters: dictionary mapping model parameter names to their
            metadata.
        """
        inner_parameters = {
            original_param: value
            for new_param, original_param in self.mapped_args.items()
            if (value := model_parameters.get(new_param)) is not None
        }

        inner_parameters.update(
            {
                param: ModelParameter(
                    lower_bound=value, upper_bound=value, fixed_to=value
                )
                for param, value in self.fixed_params.items()
            }
        )

        param_guesses = self._inner_estimate_parameters(x, y, inner_parameters)

        return {
            new_param: param_guesses[original_param]
            for new_param, original_param in self.mapped_args.items()
        }
