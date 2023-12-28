import copy
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .. import Model
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class AggregateModel(Model):
    """Model formed by aggregating one or more models.

    Aggregate models are useful for situations where one wants to analyse multiple
    models simultaneously, for example in automated tooling (e.g. ndscan
    OnlineAnalyses). In the future their functionality will be expanded to allow making
    parameters common to do joint fitting.
    """

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
          passed-in models are considered "owned" by the AggregateModel and should not
          be reused / modified externally.

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

    def can_rescale(self) -> Tuple[bool, bool]:
        rescale_x, rescale_y = zip(*[model.can_rescale() for _, model in self.models])
        return all(rescale_x), all(rescale_y)

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
    ):
        for idx, (_, model) in enumerate(self.models):
            model.estimate_parameters(x, y[idx])

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

    Repeated models allow multiple datasets to be analysed simultaneously. This is
    useful, for example, when doing joint fits to datasets (using common parameters)
    or in automated tooling (for example ndscan OnlineAnalyses) which needs a single
    model for an entire dataset.
    """

    def __init__(
        self,
        inner: Model,
        common_params: Optional[List[str]] = None,
        num_repetitions: int = 2,
    ):
        """
        :param inner: The wrapped model, the implementation of `inner` will be used to
          generate data for the y channels. This model is considered owned by the
          RepeatedModel and should not be used / modified elsewhere.
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

    def can_rescale(self) -> Tuple[bool, bool]:
        return self.inner.can_rescale()

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
    ):
        dim = self.inner.get_num_y_channels()

        common_params = {param: self.parameters[param] for param in self.common_params}
        common_heuristics = {param: [] for param in self.common_params}

        # FIXME - add "clear heuristics method instead"
        inner_params = self.inner.parameters  # store for later

        for idx in range(self.num_repetitions):
            params = {
                param: self.parameters[f"{param}_{idx}"]
                for param in self.independent_params
            }

            params.update(common_params)

            # Since common params are reused multiple times, clear their heuristic
            # values between each use
            for param_data in common_params.values():
                param_data.heuristic = None

            self.inner.parameters = params
            self.inner.estimate_parameters(x, y[idx * dim : (idx + 1) * dim])

            for param in self.common_params:
                common_heuristics[param].append(params[param].get_initial_value())

        for param in self.common_params:
            self.parameters[param].heuristic = np.mean(common_heuristics[param])

        self.inner.parameters = inner_params

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
                param_values = np.array(
                    [
                        values[f"{param_name}_{idx}"]
                        for idx in range(self.num_repetitions)
                    ]
                )
                param_uncerts = np.array(
                    [
                        uncertainties[f"{param_name}_{idx}"]
                        for idx in range(self.num_repetitions)
                    ]
                )

                derived_params[f"{param_name}_mean"] = np.mean(param_values)
                derived_uncertainties[f"{param_name}_mean"] = (
                    np.sqrt(np.sum(param_uncerts**2)) / self.num_repetitions
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
        wrapped_model: Model,
        param_mapping: Dict[str, str],
        fixed_params: Optional[Dict[str, float]] = None,
    ):
        """Init

        :param wrapped_model: The wrapped model. This model is considered "owned" by the
            MappedModel and should not be modified / used elsewhere.
        :param param_mapping: dictionary mapping names of parameters in the new
            model to names of parameters used in the wrapped model.
        :param fixed_params: dictionary mapping names of parameters used in the
            wrapped model to values they are fixed to in the new model. These
            will not be parameters of the new model.
        """
        self.wrapped_model = wrapped_model
        wrapped_params = self.wrapped_model.parameters

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
            param_name: self.wrapped_model.parameters[param_name]
            for param_name in fixed_params.keys()
        }
        for param_name, fixed_to in fixed_params.items():
            self.fixed_params[param_name].fixed_to = fixed_to

        self.param_mapping = param_mapping
        exposed_params = {
            new_name: self.wrapped_model.parameters[old_name]
            for new_name, old_name in param_mapping.items()
        }
        internal_parameters = (
            list(self.fixed_params.values()) + self.wrapped_model.internal_parameters
        )
        super().__init__(
            parameters=exposed_params,
            internal_parameters=internal_parameters,
        )

    def get_num_y_channels(self) -> int:
        return self.wrapped_model.get_num_y_channels()

    def can_rescale(self) -> Tuple[bool, bool]:
        return self.wrapped_model.can_rescale()

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
        return self.wrapped_model.func(x, new_params)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
    ):
        self.wrapped_model.estimate_parameters(x, y)
