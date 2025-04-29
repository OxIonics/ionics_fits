from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...common import TX, TY, Model
from ...utils import TX_SCALE, TY_SCALE
from ..utils import param_like


class RepeatedModel(Model):
    """Model formed by repeating a :class:`~ionics_fits.common.Model` one or more models
    along the y-axis to produce a new model, with greater y-axis dimensionality.

    The number of y-axis dimensions in the ``RepeatedModel`` is the product of the
    number of y-axis dimensions form the original model and the number of repetitions.

    Repeated models allow multiple datasets to be analysed simultaneously. This is
    useful, for example, when doing joint fits to datasets (using common parameters).

    The ``RepeatedModel`` has the following parameters and fit results:

    * all common parameters of the repeated model are parameters of the new model
    * for each independent (not common) parameter of the repeated model, ``param``,
      the ``RepeatedModel`` has parameters ``param_{n}`` for ``n`` in
      ``[0, .., num_repitions-1]``
    * for each independent parameter, ``param``, the ``RepeatedModel`` model
      has additional fit results representing statistics between the repetitions.
      For each independent parameter ``param``, the results dictionary will have
      additional quantities ``param_mean`` and ``param_peak_peak``.

    If ``aggregate_results`` is enabled the rules for fit results are modified as
    follows:

    * no statistical quantities are calculated for fit parameters or derived results
    * all derived results whose values and uncertainties are the same for all
      repetitions are aggregated together to give a single result. These results have
      the same name as the original model's results, with no suffix.
    * all derived results whose value is not the same for al repetitions are omitted.
    """

    def __init__(
        self,
        model: Model,
        common_params: Optional[List[str]] = None,
        num_repetitions: int = 2,
        aggregate_results=False,
    ):
        """
        :param model: The :class:`~ionics_fits.common.Model` to be repeated. The
          implementation of ``model`` will be used to generate data for all y axes. This
          model is considered owned by the ``RepeatedModel`` and should not be used /
          modified elsewhere.
        :param common_params: optional list of names of model parameters, whose value is
          common to all y axes. All other model parameters are allowed to vary
          independently between the y axes
        :param num_repetitions: the number of times the model is to be repeated
        :param aggregate_results: determines whether derived results are aggregated or
          not. The default behaviour is to not aggregate results. This is
          generally suitable when one wants access to the values of non-common
          parameters from the various repetitions. Aggregating results can be useful,
          for example, when all parameters are common across the repetitions and one
          wants a single set of values reported.
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

        def wrapped_scale_func(
            scale_func: Callable, x_scales: TX_SCALE, y_scales: TY_SCALE
        ) -> float:
            return scale_func(x_scales, y_scales[0 : self.model.get_num_y_axes()])

        parameters = list(self.parameters.values()) + self.internal_parameters
        for parameter in parameters:
            scale_func = parameter.scale_func
            parameter.scale_func = partial(wrapped_scale_func, scale_func)
            parameter.scale_func.__name__ = scale_func.__name__

    def get_num_x_axes(self) -> int:
        return self.model.get_num_x_axes()

    def get_num_y_axes(self) -> int:
        return self.num_repetitions * self.model.get_num_y_axes()

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        # Disable y-axis rescaling so we don't end up with multiple different scale
        # factors for the common parameters. To support rescaling here we would need to
        # extend Fitter to allow the model to have some control over the axis scale
        # factors.
        x_scales, _ = self.model.can_rescale()
        return x_scales, [False] * self.get_num_y_axes()

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        x = np.atleast_2d(x)
        values = {param: param_values[param] for param in self.common_params}

        dim = self.model.get_num_y_axes()
        num_samples = x.shape[1]
        ys = np.zeros((self.get_num_y_axes(), num_samples))
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

    def estimate_parameters(self, x: TX, y: TY):
        dim = self.model.get_num_y_axes()

        common_heuristics = {
            param: np.zeros(self.num_repetitions + 1) for param in self.common_params
        }

        for idx in range(self.num_repetitions):
            params = {
                param: self.parameters[f"{param}_{idx}"]
                for param in self.independent_params
            }

            params.update(
                {param: self.parameters[param] for param in self.common_params}
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
            common_heuristics[param][-1] = np.mean(common_heuristics[param][:-1])

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
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        derived_params = {}
        derived_uncertainties = {}

        dim = self.model.get_num_y_axes()

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
