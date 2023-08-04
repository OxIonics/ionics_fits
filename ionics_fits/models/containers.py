import copy
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
          parameter `bar`, the aggregate model will have a parameter `foo_bar`.

        At present this class only supports models with a single y channel. This is just
        because no one got around to implementing it yet rather than any fundamental
        difficulty.

        A possible future extension of this class would be to introduce a notion of
        "common parameters", whose value is the same for all of the aggregated models. A
        design question here is where the common parameters should get their properties
        (bounds, scale factors, etc) from and how we should deal with their heuristics.
        """
        self.models = copy.deepcopy(models)

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
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        ys = []
        for model_name, model in self.models:
            model_params = {
                param_name: param_values[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }
            ys.append(model.func(x, model_params))

        return np.array(ys)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        for idx, (model_name, model) in enumerate(self.models):
            params = {
                param_name: model_parameters[f"{model_name}_{param_name}"]
                for param_name in model.parameters.keys()
            }
            model.estimate_parameters(x, y[idx], params)

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
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
                y=y[idx],
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


class RepeatedModel(Model):
    """Model formed by repeating a `Model` one more times

    The `RepeatedModel` has multiple y-channels, corresponding to the repetitions of
    the wrapped model.
    """

    def __init__(
        self,
        inner: Model,
        common_params: Optional[List[str]] = None,
        num_repetitions: int = 2,
    ):
        """
        :param inner: The wrapped model, the implementation of `inner` will be used to
          generate data for the y channels
        :param common_params: optional list of names of model arguments, whose value is
          common to all y channels. All other model parameters are independent
        :param num_repetitions: the number of times the inner model is repeated

        Parameters of the new model:
          - all common parameters of the inner model are parameters of the outer model
          - for each independent (not common) or derived parameter of the inner model,
            `foo`, the outer model has parameters `foo_{n}` for n in
            [0, .., num_repitions-1]
          - for each independent or derived parameter, `foo`, the outer model calculates
            statistics for the results from the various models: `foo_mean` and
            `foo_peak_peak`.
        """
        inner = copy.deepcopy(inner)

        inner_params = set(inner.parameters.keys())
        common_params = set(common_params or [])

        if not common_params.issubset(inner_params):
            raise ValueError(
                "Common parameters must be a subset of the inner model's parameters"
            )

        independent_params = set(inner.parameters.keys()) - common_params
        params = {param: inner.parameters[param] for param in common_params}
        for param in independent_params:
            params.update(
                {
                    f"{param}_{idx}": copy.deepcopy(inner.parameters[param])
                    for idx in range(num_repetitions)
                }
            )

        super().__init__(parameters=params)

        self.inner = inner
        self.common_params = common_params
        self.independent_params = independent_params
        self.num_repetitions = num_repetitions

    def get_num_y_channels(self) -> int:
        return self.num_repetitions * self.inner.get_num_y_channels()

    def func(
        self,
        x: Array[("num_samples",), np.float64],
        param_values: Dict[str, float],
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        common_values = {param: param_values[param] for param in self.common_params}

        ys = []
        for idx in range(self.num_repetitions):
            values = dict(common_values)
            values.update(
                {
                    param: param_values[f"{param}_{idx}"]
                    for param in self.independent_params
                }
            )
            ys.append(np.atleast_2d(self.inner.func(x, values)))

        return np.vstack(ys)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        dim = self.inner.get_num_y_channels()

        common_params = {param: model_parameters[param] for param in self.common_params}
        common_heuristics = {param: [] for param in self.common_params}
        for idx in range(self.num_repetitions):
            params = {
                param: model_parameters[f"{param}_{idx}"]
                for param in self.independent_params
            }
            params.update(copy.deepcopy(common_params))
            self.inner.estimate_parameters(x, y[idx * dim : (idx + 1) * dim], params)

            for param in self.common_params:
                common_heuristics[param].append(params[param].get_initial_value())

        for param in self.common_params:
            model_parameters[param].heuristic = np.mean(common_heuristics[param])

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        dim = self.inner.get_num_y_channels()

        for idx in range(self.num_repetitions):
            rep_params = {
                param: fitted_params[f"{param}_{idx}"]
                for param in self.independent_params
            }
            rep_params.update(
                {param: fitted_params[param] for param in self.common_params}
            )

            rep_uncertainties = {
                param: fit_uncertainties[f"{param}_{idx}"]
                for param in self.independent_params
            }
            rep_uncertainties.update(
                {param: fit_uncertainties[param] for param in self.common_params}
            )

            derived = self.inner.calculate_derived_params(
                x=x,
                y=y[idx * dim : (idx + 1) * dim],
                fitted_params=rep_params,
                fit_uncertainties=rep_uncertainties,
            )
            rep_derived_params, rep_derived_uncertainties = derived

            derived_params.update(
                {f"{param}_{idx}": value for param, value in rep_derived_params.items()}
            )
            derived_uncertainties.update(
                {
                    f"{param}_{idx}": value
                    for param, value in rep_derived_uncertainties.items()
                }
            )

        # calculate statistical results
        def add_statistics(values, uncertainties, param_names):
            for param_name in param_names:
                param_values = [
                    values[f"{param_name}_{idx}"] for idx in range(self.num_repetitions)
                ]
                param_uncerts = [
                    uncertainties[f"{param_name}_{idx}"]
                    for idx in range(self.num_repetitions)
                ]

                derived_params[f"{param_name}_mean"] = np.mean(param_values)
                derived_uncertainties[f"{param_name}_mean"] = (
                    np.sqrt(np.sum(np.power(param_uncerts, 2))) / self.num_repetitions
                )

                derived_params[f"{param_name}_peak_peak"] = np.ptp(param_values)
                derived_uncertainties[f"{param_name}_peak_peak"] = float("nan")

        add_statistics(
            derived_params, derived_uncertainties, param_names=rep_derived_params.keys()
        )
        add_statistics(
            fitted_params,
            fit_uncertainties,
            param_names=self.independent_params,
        )

        return derived_params, derived_uncertainties


class MappedModel(Model):
    """`Model` wrapping another `Model` with renamed parameters"""

    def __init__(
        self,
        inner: Model,
        mapped_params: Dict[str, str],
        fixed_params: Optional[Dict[str, float]] = None,
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

        if fixed_params is None:
            fixed_params = {}

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

    def can_rescale(self, x_scale: float, y_scale: float) -> bool:
        """Returns True if the model can be rescaled"""
        return self.inner.can_rescale(x_scale, y_scale)

    @staticmethod
    def get_scaled_model(model, x_scale: float, y_scale: float):
        """Returns a scaled copy of a given model object

        :param model: model to be copied and rescaled
        :param x_scale: x-axis scale factor
        :param y_scale: y-axis scale factor
        :returns: a scaled copy of model
        """
        scaled_model = copy.deepcopy(model)
        for param_name, param in scaled_model.inner.parameters.items():
            param.rescale(x_scale, y_scale, scaled_model.inner)

        for fixed_param in scaled_model.fixed_params.keys():
            scale_factor = scaled_model.inner.parameters[fixed_param].scale_factor
            scaled_model.fixed_params[fixed_param] /= scale_factor

        # Expose the scale factors to the fitter so it knows how to rescale the results
        for new_name, old_name in scaled_model.mapped_args.items():
            scale_factor = scaled_model.inner.parameters[old_name].scale_factor
            scaled_model.parameters[new_name].scale_factor = scale_factor

        return scaled_model

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return self.inner.get_num_y_channels()

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples", "num_y_channels"), np.float64]:
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
        y: Array[("num_samples", "num_y_channels"), np.float64],
        inner_parameters: Dict[str, ModelParameter],
    ) -> Dict[str, float]:
        return self.inner.estimate_parameters(x, y, inner_parameters)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        model_parameters: Dict[str, ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        inner_parameters = {
            original_param: copy.deepcopy(model_parameters[new_param])
            for new_param, original_param in self.mapped_args.items()
        }

        inner_parameters.update(
            {
                param: ModelParameter(
                    lower_bound=value, upper_bound=value, fixed_to=value
                )
                for param, value in self.fixed_params.items()
            }
        )

        self._inner_estimate_parameters(x, y, inner_parameters)

        for new_param, original_param in self.mapped_args.items():
            initial_value = inner_parameters[original_param].get_initial_value()
            model_parameters[new_param].heuristic = initial_value
