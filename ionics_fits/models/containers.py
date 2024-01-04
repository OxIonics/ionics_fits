from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .utils import param_like
from .. import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class AggregateModel(Model):
    """Model formed by aggregating one or more models.

    When aggregating a number of identical models, use a :class RepeatedModel: instead.

    Aggregate models have a number of uses. For example:
      - joint fits to multiple data sets (pass the datasets in as y channels and use
        "common" parameters for any parameters which are fit jointly to all y channels).
      - fit multiple datasets simultaneously. This is useful, for example in automated
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

        At present this class only supports models with a single y channel. This is just
        because no one got around to implementing it yet rather than any fundamental
        difficulty.
        """
        self.models = models
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
            if model.get_num_y_channels() != 1:
                raise ValueError(
                    "AggregateModel currently only supports models with a single y "
                    "channel"
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

    def get_num_y_channels(self) -> int:
        return len(self.models)

    def can_rescale(self) -> Tuple[bool, bool]:
        rescale_x, rescale_y = zip(
            *[model.can_rescale() for model in self.models.values()]
        )
        return all(rescale_x), all(rescale_y)

    def func(
        self,
        x: Array[("num_samples",), np.float64],
        param_values: Dict[str, float],
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        ys = np.zeros((self.get_num_y_channels(), len(x)), dtype=np.float64)
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

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
    ):
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
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
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


class RepeatedModel(Model):
    """Model formed by repeating a `Model` one more times.

    The `RepeatedModel` has multiple y-channels, corresponding to the repetitions of
    the repeated model.

    Repeated models allow multiple datasets to be analysed simultaneously. This is
    useful, for example, when doing joint fits to datasets (using common parameters)
    or in automated tooling (for example ndscan OnlineAnalyses) which needs a single
    model for an entire dataset.
    """

    def __init__(
        self,
        model: Model,
        common_params: Optional[List[str]] = None,
        num_repetitions: int = 2,
        aggregate_results=False,
    ):
        """
        :param model: The repeated model, the implementation of `model` will be used to
          generate data for all y channels. This model is considered owned by the
          RepeatedModel and should not be used / modified elsewhere.
        :param common_params: optional list of names of model parameters, whose value is
          common to all y channels. All other model parameters are allowed to vary
          independently between the y channels
        :param num_repetitions: the number of times the model is repeated
        :param aggregate_results: determines whether derived results are aggregated or
          not (see below). The default behaviour is to not aggregate results. This is
          generally suitable when one wants access to the values of non-common
          parameters from the various repetitions. Aggregating results can be useful,
          for example, when all parameters are common across the repetitions and one
          wants a single set of values reported.

        The `RepeatedModel` has the following parameters and fit results:
          - all common parameters of the repeated model are parameters of the new model
          - for each independent (not common) parameter of the repeated model, `param`,
            the `RepeatedModel` has parameters `param_{n}` for n in
            [0, .., num_repitions-1]
          - for each independent parameter, `param`, the `RepeatedModel` model
            has additional fit results representing statistics between the repetitions.
            For each independent parameter `param`, the results dictionary will have
            additional quantities: `param_mean` and `param_peak_peak`.

        If :param aggregate_results: is `False` the rules for fit results follow those
        described previously. This is the default behaviour.

        If `aggregate_results` is `True`:
          - no statistical quantities are calculated for fit parameters or derived
            results
          - all derived results whose values and uncertainties are the same for all
            repetitions are aggregated together to give a single result. This has the
            same name as the original model (no suffix).
          - all derived results whose value is not the same for al repetitions are
            omitted.
        """
        self.aggregate_results = aggregate_results
        model_params = set(model.parameters.keys())
        common_params = set(common_params or [])

        if not common_params.issubset(model_params):
            raise ValueError(
                "Common parameters must be a subset of the model's parameters"
            )

        independent_params = set(model_params) - common_params
        params = {param: model.parameters[param] for param in common_params}
        for param in independent_params:
            params.update(
                {
                    f"{param}_{idx}": param_like(model.parameters[param])
                    for idx in range(num_repetitions)
                }
            )

        self.model = model
        self.common_params = common_params
        self.independent_params = independent_params
        self.num_repetitions = num_repetitions

        super().__init__(
            parameters=params, internal_parameters=self.model.internal_parameters
        )

    def get_num_y_channels(self) -> int:
        return self.num_repetitions * self.model.get_num_y_channels()

    def can_rescale(self) -> Tuple[bool, bool]:
        return self.model.can_rescale()

    def func(
        self,
        x: Array[("num_samples",), np.float64],
        param_values: Dict[str, float],
    ) -> Array[("num_y_channels", "num_samples"), np.float64]:
        values = {param: param_values[param] for param in self.common_params}

        dim = self.model.get_num_y_channels()
        ys = np.zeros((self.get_num_y_channels(), len(x)))
        for idx in range(self.num_repetitions):
            values.update(
                {
                    param: param_values[f"{param}_{idx}"]
                    for param in self.independent_params
                }
            )
            ys[idx * dim : (idx + 1) * dim, :] = np.atleast_2d(
                self.model.func(x, values)
            )

        return ys

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
    ):
        dim = self.model.get_num_y_channels()

        common_heuristics = {
            param: [0.0] * self.num_repetitions for param in self.common_params
        }

        for idx in range(self.num_repetitions):
            params = {
                param: self.parameters[f"{param}_{idx}"]
                for param in self.independent_params
            }

            params.update(
                {
                    param: self.parameters[param]
                    for param in self.common_params
                }
            )

            # Reset heuristics before each iteration
            for param_data in params.values():
                param_data.heuristic = None

            self.model.parameters = params
            self.model.estimate_parameters(x, y[idx * dim : (idx + 1) * dim])

            for param in self.common_params:
                common_heuristics[param][idx] = params[param].get_initial_value()

        # Combine the heuristics for the repetitions to find the best set of common
        # parameter values
        for param in self.common_params:
            common_heuristics[param].append(np.mean(common_heuristics[param]))

        param_estimates = {
            param_name: self.parameters[param_name].get_initial_value()
            for param_name in self.parameters.keys()
            if param_name not in self.common_params
        }
        costs = np.zeros(self.num_repetitions + 1)
        for idx in range(self.num_repetitions + 1):
            y_idx = self.__call__(
                x=x,
                **param_estimates,
                **{
                    param_name: common_heuristics[param_name][idx]
                    for param_name in self.common_params
                },
            )
            costs[idx] = np.sqrt(np.sum((y - y_idx) ** 2))
        best_heuristic = np.argmin(costs)

        for param in self.common_params:
            self.parameters[param].heuristic = common_heuristics[param][best_heuristic]

    def calculate_derived_params(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        dim = self.model.get_num_y_channels()

        # {param_name: [derived_values_from_each_repetition]}
        derived_params_reps = None
        derived_uncertainties_reps = None
        derived_param_names = None

        common_fitted_params = {
            param: fitted_params[param] for param in self.common_params
        }
        common_fit_uncertainties = {
            param: fit_uncertainties[param] for param in self.common_params
        }

        for idx in range(self.num_repetitions):
            rep_params = {
                param: fitted_params[f"{param}_{idx}"]
                for param in self.independent_params
            }
            rep_params.update(common_fitted_params)

            rep_uncertainties = {
                param: fit_uncertainties[f"{param}_{idx}"]
                for param in self.independent_params
            }
            rep_uncertainties.update(common_fit_uncertainties)

            derived = self.model.calculate_derived_params(
                x=x,
                y=y[idx * dim : (idx + 1) * dim],
                fitted_params=rep_params,
                fit_uncertainties=rep_uncertainties,
            )
            rep_derived_params, rep_derived_uncertainties = derived

            if derived_params_reps is None:
                derived_param_names = rep_derived_params.keys()
                # We can't preallocate prior to this because we don't know what the
                # model's derived parameter values are until we've calculated them
                derived_params_reps = {
                    param_name: [0.0] * self.num_repetitions
                    for param_name in derived_param_names
                }
                derived_uncertainties_reps = {
                    param_name: [0.0] * self.num_repetitions
                    for param_name in derived_param_names
                }

            for param_name in derived_param_names:
                derived_params_reps[param_name][idx] = rep_derived_params[param_name]
                derived_uncertainties_reps[param_name][idx] = rep_derived_uncertainties[
                    param_name
                ]

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

        if self.aggregate_results:
            keep = [
                param_name
                for param_name in derived_param_names
                if np.all(
                    np.isclose(
                        derived_params_reps[param_name],
                        derived_params_reps[param_name][0],
                    )
                )
                and np.all(
                    np.isclose(
                        derived_uncertainties_reps[param_name],
                        derived_uncertainties_reps[param_name][0],
                    )
                )
            ]
            derived_params.update(
                {param_name: derived_params_reps[param_name][0] for param_name in keep}
            )
            derived_uncertainties.update(
                {
                    param_name: derived_uncertainties_reps[param_name][0]
                    for param_name in keep
                }
            )
        else:
            for param_name in derived_param_names:
                derived_params.update(
                    {
                        f"{param_name}_{idx}": derived_params_reps[param_name][idx]
                        for idx in range(self.num_repetitions)
                    }
                )
                derived_uncertainties.update(
                    {
                        f"{param_name}_{idx}": derived_uncertainties_reps[param_name][
                            idx
                        ]
                        for idx in range(self.num_repetitions)
                    }
                )

            add_statistics(
                derived_params,
                derived_uncertainties,
                param_names=rep_derived_params.keys(),
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

        for param_name, param_data in self.parameters.items():
            old_name = self.param_mapping[param_name]
            old_param = self.wrapped_model.parameters[old_name]
            param_data.heuristic = old_param.heuristic