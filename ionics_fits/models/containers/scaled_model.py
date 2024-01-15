from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from ... import Model, common
from ...utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class ScaledModel(Model):
    """Model with rescaled x-axis.

    This is useful, for example, to convert models between linear and angular units.
    """

    def __init__(self, model: Model, x_scale: float):
        """
        :param model: model to rescale. This model is considered "owned" by the
          :class ScaledModel: and should not be used/modified elsewhere.
        :param x_scale: multiplicative x-axis scale factor. To convert a model that
          takes x in angular units and convert to one that takes x in linear units use
          `x_scale = 2 * np.pi`
        """
        self.model = model
        self.x_scale = x_scale
        self._rescale = True

        super().__init__(
            parameters=self.model.parameters,
            internal_parameters=self.model.internal_parameters,
        )

    def func(
        self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
    ) -> Array[("num_samples", "num_y_channels"), np.float64]:
        x = (x * self.x_scale) if self._rescale else x
        return self.model.func(x, param_values)

    def estimate_parameters(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
    ):
        # avoid double rescaling if estimate_parameters calls self.func internally
        self._rescale = False
        super().estimate_parameters(x * self._x_scale, y)
        self._rescale = True

    def can_rescale(self) -> Tuple[bool, bool]:
        return self.model.can_rescale()

    def get_num_y_channels(self) -> int:
        return self.model.get_num_y_channels()

    def calculate_derived_params(
        self,
        x: common.TX,
        y: common.TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        return self.model.calculate_derived_params()
