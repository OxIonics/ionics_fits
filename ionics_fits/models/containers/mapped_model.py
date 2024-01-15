from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ... import Model
from ...utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class MappedModel(Model):
    """`Model` wrapping another `Model` with renamed parameters"""

    def __init__(
        self,
        model: Model,
        param_mapping: Dict[str, str],
        fixed_params: Optional[Dict[str, float]] = None,
        derived_result_mapping: Optional[Dict[str, Optional[str]]] = None,
    ):
        """Init

        :param model: The wrapped model. This model is considered "owned" by the
            MappedModel and should not be modified / used elsewhere.
        :param param_mapping: dictionary mapping names of parameters in the new
            model to names of parameters used in the wrapped model.
        :param fixed_params: dictionary mapping names of parameters used in the
            wrapped model to values they are fixed to in the new model. These
            will not be parameters of the new model.
        :param derived_result_mapping: optional dictionary mapping names of derived
            result in the new model to names of derived results in the wrapped model.
            Derived results may be renamed to `None` to exclude them from the model's
            outputs.
        """
        self.model = model
        self.derived_result_mapping = derived_result_mapping or {}
        wrapped_params = self.model.parameters

        fixed_params = fixed_params or {}

        if unknown_mapped_params := set(param_mapping.values()) - wrapped_params.keys():
            raise ValueError(
                "The target of parameter mappings must be parameters of the inner "
                f"model. The following mapping targets are not: {unknown_mapped_params}"
            )

        if unknown_fixed_params := fixed_params.keys() - wrapped_params.keys():
            raise ValueError(
                "Fixed parameters must be parameters of the inner model. The "
                f"follow fixed parameters are not: {unknown_fixed_params}"
            )

        if missing_params := wrapped_params.keys() - (
            fixed_params.keys() | param_mapping.values()
        ):
            raise ValueError(
                "All parameters of the inner model must be either mapped of "
                "fixed. The following inner model parameters are neither: "
                f"{missing_params}"
            )

        if duplicated_params := fixed_params.keys() & param_mapping.values():
            raise ValueError(
                "Parameters cannot be both mapped and fixed. The following "
                f"parameters are both: {duplicated_params}"
            )

        self.fixed_params = {
            param_name: self.model.parameters[param_name]
            for param_name in fixed_params.keys()
        }
        for param_name, fixed_to in fixed_params.items():
            self.fixed_params[param_name].fixed_to = fixed_to

        self.param_mapping = param_mapping
        exposed_params = {
            new_name: self.model.parameters[old_name]
            for new_name, old_name in param_mapping.items()
        }
        internal_parameters = (
            list(self.fixed_params.values()) + self.model.internal_parameters
        )

        super().__init__(
            parameters=exposed_params,
            internal_parameters=internal_parameters,
        )

    def get_num_y_channels(self) -> int:
        return self.model.get_num_y_channels()

    def can_rescale(self) -> Tuple[bool, bool]:
        return self.model.can_rescale()

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples", "num_y_channels"), np.float64]:
        new_params = {
            old_name: param_values[new_name]
            for new_name, old_name in self.param_mapping.items()
        }
        new_params.update(
            {
                param_name: param_data.fixed_to
                for param_name, param_data in self.fixed_params.items()
            }
        )
        return self.model.func(x, new_params)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
    ):
        self.model.estimate_parameters(x, y)

        for param_name, param_data in self.parameters.items():
            old_name = self.param_mapping[param_name]
            old_param = self.model.parameters[old_name]
            param_data.heuristic = old_param.heuristic

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        new_fitted_params = {
            old_name: fitted_params[new_name]
            for new_name, old_name in self.param_mapping.items()
        }
        new_fitted_params.update(
            {
                param_name: param_data.fixed_to
                for param_name, param_data in self.fixed_params.items()
            }
        )

        new_fit_uncertainties = {
            old_name: fit_uncertainties[new_name]
            for new_name, old_name in self.param_mapping.items()
        }
        new_fit_uncertainties.update(
            {param_name: 0.0 for param_name, param_data in self.fixed_params.items()}
        )
        derived = self.model.calculate_derived_params(
            x=y,
            y=y,
            fitted_params=new_fitted_params,
            fit_uncertainties=new_fit_uncertainties,
        )
        mapping = self.derived_result_mapping
        for derived_dict in derived:
            for new_result_name, model_result_name in mapping.items():
                if new_result_name is None:
                    del derived_dict[model_result_name]
                    continue

                if model_result_name not in derived_dict:
                    raise ValueError(
                        f"Mapped derived result '{model_result_name}' not found."
                    )
                value = derived_dict.pop(model_result_name)
                if new_result_name in derived_dict:
                    raise ValueError(
                        f"Mapped derived result '{new_result_name}' duplicates "
                        "existing derived resut name."
                    )
                derived_dict[new_result_name] = value

        return derived