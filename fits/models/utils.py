from typing import Dict, Tuple

import numpy as np

from ..common import FitModel
from ..utils import Array


class MappedFitModel(FitModel):
    def __init__(
        self,
        inner: FitModel,
        mapped_params: Dict[str, str],
        fixed_params: Dict[str, float] = None,
    ):
        inner_params = inner.get_parameters()

        if not set(mapped_params.values()).union(set(fixed_params)) == set(
            inner_params
        ):
            raise ValueError(
                "Parameter map does not match original model class parameters"
            )

        if not set(fixed_params).issubset(set(inner_params)):
            raise ValueError("Fixed parameters must be parameters of inner model class")

        if set(fixed_params).intersection(set(mapped_params.values())):
            raise ValueError("Fixed parameters must not feature in the parameter map")

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
