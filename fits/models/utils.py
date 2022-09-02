from typing import Dict, Tuple

import numpy as np

from ..common import FitModel
from ..utils import Array


class MappedFitModel(FitModel):
    """`FitModel` wrapping another `FitModel` with renamed parameters"""

    def __init__(
        self,
        inner: FitModel,
        mapped_params: Dict[str, str],
        fixed_params: Dict[str, float] = None,
    ):
        """Init

        :param inner: The wrapped fit model, the implementation of `inner` will
            be used after the parameter mapping has been done.
        :param mapped_params: dictionary mapping names of parameters in the new
            model to names of parameters used in the wrapped model.
        :param fixed_params: dictionary mapping names of parameters used in the
            wrapped model to values they are fixed to in the new model. These
            will not be parameters of the new model.
        """
        inner_params = inner.get_parameters()

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
        self, x: Array[("num_samples",), np.float64], params: Dict[str, float]
    ) -> Array[("num_samples",), np.float64]:
        new_params = {
            old_name: params[new_name]
            for new_name, old_name in self.mapped_args.items()
        }
        new_params.update(self.fixed_params)
        return self.inner.func(x, new_params)

    def _inner_estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ):
        return self.inner.estimate_parameters(x, y, known_values, bounds)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        known_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        known_values = {
            original_param: value
            for new_param, original_param in self.mapped_args.items()
            if (value := known_values.get(original_param)) is not None
        }
        known_values.update(self.fixed_params)

        bounds = {
            self.mapped_args[new_param]: bounds for new_param, bounds in bounds.items()
        }
        bounds.update(
            {
                original_param: (value, value)
                for original_param, value in self.fixed_params.items()
            }
        )

        param_guesses = self._inner_estimate_parameters(x, y, known_values, bounds)

        return {
            new_param: param_guesses[original_param]
            for new_param, original_param in self.mapped_args.items()
        }
