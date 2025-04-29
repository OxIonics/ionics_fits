from typing import Dict, List, Tuple

from ...common import TX, TY, Model, ModelParameter


class ReparametrizedModel(Model):
    r"""Model formed by reparametrizing an existing :class:`~ionics_fits.common.Model`.

    ``ionics_fits`` aims to provide convenient and flexible model parametrisations,
    however sometimes the default parametrisation won't be convenient for your
    application. For these cases ``ReparametrizedModel``\s provide a convenient way of
    changing and extending the parametrisation of an existing model.

    Reparametrizing a model involves replacing "bound" parameters with "new" parameters,
    whose values the bound parameter values are calculated from.

    All non-bound parameters of the original model as well as all "new" parameters are
    parameters of the :class:`ReparametrizedModel` (bound parameters are internal
    parameters of the new model, but are not directly exposed to the user). All derived
    results from the original model are derived results from the
    :class:`ReparametrizedModel` (override ``calculate_derived_params`` to change
    this behaviour).

    Values and uncertainties for the bound parameters are exposed as derived results.

    Subclasses must override :meth:`bound_param_values`,
    :meth:`bound_param_uncertainties` and :meth:`bound_param_uncertainties` to specify
    the mapping between "new" and "bound" parameters.

    Example usage converting a sinusoid parameterised by offset and amplitude into one
    parametrised by minimum and maximum values:

    .. testcode::

        from typing import Dict

        from ionics_fits.models.sinusoid import Sinusoid
        from ionics_fits.models.transformations.reparametrized_model import (
            ReparametrizedModel
        )

        class SineMinMax(ReparametrizedModel):
            def __init__(self):
                super().__init__(
                    model=Sinusoid(),
                    new_params={
                        "min": ModelParameter(scale_func=scale_y()),
                        "max": ModelParameter(scale_func=scale_y()),
                    },
                    bound_params=["a", "y0"],
                )

            @staticmethod
            def bound_param_values(param_values: Dict[str, float]) -> Dict[str, float]:
                return {
                    "a": 0.5 * (param_values["max"] - param_values["min"]),
                    "y0": 0.5 * (param_values["max"] + param_values["min"]),
                }

            @staticmethod
            def bound_param_uncertainties(
                param_values: Dict[str, float], param_uncertainties: Dict[str, float]
            ) -> Dict[str, float]:
                err = 0.5 * np.sqrt(
                    param_uncertainties["max"] ** 2 + param_uncertainties["min"] ** 2
                )
                return {"a": err, "y0": err}

            @staticmethod
            def new_param_values(model_param_values: Dict[str, float]
            ) -> Dict[str, float]:
                return {
                    "max": (model_param_values["y0"] + model_param_values["a"]),
                    "min": (model_param_values["y0"] - model_param_values["a"]),
                }

    See also :class:`~ionics_fits.models.transformations.mapped_model.MappedModel`.
    """

    def __init__(
        self,
        model: Model,
        new_params: Dict[str, ModelParameter],
        bound_params: List[str],
    ):
        """
        :param model: The model to be reparametrized. This model is considered "owned"
          by the ``ReparametrizedModel`` and should not be used / modified elsewhere.
        :param new_params: dictionary of new parameters of the ``ReparametrizedModel``
        :param bound_params: list of parameters of the
          :class:`~ionics_fits.common.Model` to bound. These parameters are
          not exposed as parameters of the ``ReparametrizedModel``.
        """
        self.model = model
        self.bound_params = bound_params
        self.unbound_model_params = [
            param_name
            for param_name in self.model.parameters.keys()
            if param_name not in bound_params
        ]

        internal_parameters = self.model.internal_parameters
        parameters = dict(self.model.parameters)

        if not set(bound_params).issubset(set(parameters.keys())):
            raise ValueError("Bound parameters must all be parameters of the model")

        for param_name in bound_params:
            internal_parameters.append(parameters.pop(param_name))

        duplicates = set(new_params.keys()).intersection(parameters.keys())
        if duplicates:
            raise ValueError(
                "New parameter names must not duplicate names of unbound model "
                f"parametes. Duplicates are {duplicates}."
            )

        parameters.update(new_params)

        super().__init__(parameters=parameters, internal_parameters=internal_parameters)

    @staticmethod
    def bound_param_values(param_values: Dict[str, float]) -> Dict[str, float]:
        """Returns a dictionary of values of the model's bound parameters.

        This method must be overridden to specify the mapping from parameters of the
        ``ReparameterizedModel`` to values of the bound parameters.

        :param new_param_values: dictionary of parameter values for the
          ``ReparameterizedModel``.
        :returns: dictionary of values for the bound parameters of the original model.
        """
        raise NotImplementedError

    @staticmethod
    def bound_param_uncertainties(
        param_values: Dict[str, float], param_uncertainties: Dict[str, float]
    ) -> Dict[str, float]:
        """Returns a dictionary of uncertainties for the model's bound parameters.

        This method must be overridden to specify the mapping from parameter
        uncertainties for the ``ReparameterizedModel`` to bound parameter
        uncertainties.

        :param param_values: dictionary of values for parameters of the
          ``ReparameterizedModel``.
        :param param_uncertainties: dictionary of uncertainties for parameters of the
          ``ReparameterizedModel``.
        :returns: dictionary of values for the bound parameters of the original model.
        """
        raise NotImplementedError

    @staticmethod
    def new_param_values(model_param_values: Dict[str, float]) -> Dict[str, float]:
        r"""Returns a dictionary of values of the model's "new" parameters.

        This method must be overridden to specify the mapping from values of the
        original model to values of the ``ReparametrizedModel``\'s "new" parameters.

        This is used to find estimates for the new parameters from the original
        model's parameter estimates.

        :param model_param_values: dictionary of parameter values for the original model
        :returns: dictionary of values for the "new" parameters of the
          ``ReparametrizedModel``.
        """
        raise NotImplementedError

    def get_num_x_axes(self) -> int:
        return self.model.get_num_x_axes()

    def get_num_y_axes(self) -> int:
        return self.model.get_num_y_axes()

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return self.model.can_rescale()

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        model_values = self.bound_param_values(param_values)
        model_values.update(
            {
                param_name: param_values[param_name]
                for param_name in self.unbound_model_params
            }
        )
        return self.model.func(x, model_values)

    def estimate_parameters(self, x: TX, y: TX):
        self.model.estimate_parameters(x=x, y=y)
        model_heuristics = {
            param_name: param_data.get_initial_value()
            for param_name, param_data in self.model.parameters.items()
        }
        new_heuristics = self.new_param_values(model_heuristics)

        for param_name in self.unbound_model_params:
            self.parameters[param_name].heuristic = model_heuristics[param_name]

        for param_name, param_heuristic in new_heuristics.items():
            self.parameters[param_name].heuristic = param_heuristic

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        bound_values = self.bound_param_values(fitted_params)
        bound_uncertainties = self.bound_param_uncertainties(
            fitted_params, fit_uncertainties
        )

        model_fitted_params = {
            param_name: fitted_params[param_name]
            for param_name in self.unbound_model_params
        }
        model_fit_uncertainties = {
            param_name: fit_uncertainties[param_name]
            for param_name in self.unbound_model_params
        }

        model_fitted_params.update(bound_values)
        model_fit_uncertainties.update(bound_uncertainties)

        derived_values, derived_uncertainties = self.model.calculate_derived_params(
            x=y,
            y=y,
            fitted_params=model_fitted_params,
            fit_uncertainties=model_fit_uncertainties,
        )

        derived_values.update(bound_values)
        derived_uncertainties.update(bound_uncertainties)

        return derived_values, derived_uncertainties
