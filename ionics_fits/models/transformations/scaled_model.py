from typing import Dict, List, Tuple

from ...common import TX, TY, Model


class ScaledModel(Model):
    r"""Model with rescaled x-axis.

    A common use-case for ``ScaledModel``\s is converting models between linear and
    angular units.
    """

    def __init__(self, model: Model, x_scale: float, x_offset: float = 0.0):
        """
        :param model: model to rescale. This model is considered "owned" by the
          ``ScaledModel`` and should not be used/modified elsewhere.
        :param x_scale: multiplicative x-axis scale factor. To convert a model that
          takes x in angular units and convert to one that takes x in linear units use
          ``x_scale = 2 * np.pi``
        :param x_offset: additive x-axis offset
        """
        self.model = model
        self.x_scale = x_scale
        self.x_offset = x_offset

        super().__init__(
            parameters=self.model.parameters,
            internal_parameters=self.model.internal_parameters,
        )

    def func(self, x: TX, param_values: Dict[str, float]) -> TY:
        return self.model.func(x * self.x_scale + self.x_offset, param_values)

    def estimate_parameters(self, x: TX, y: TY):
        self.model.estimate_parameters(x * self.x_scale + self.x_offset, y)
        for param_name, param_data in self.model.parameters.items():
            self.parameters[param_name].heuristic = param_data.get_initial_value()

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return self.model.can_rescale()

    def get_num_x_axes(self) -> int:
        return self.model.get_num_x_axes()

    def get_num_y_axes(self) -> int:
        return self.model.get_num_y_axes()

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        return self.model.calculate_derived_params(
            x=x * self.x_scale + self.x_offset,
            y=y,
            fitted_params=fitted_params,
            fit_uncertainties=fit_uncertainties,
        )
