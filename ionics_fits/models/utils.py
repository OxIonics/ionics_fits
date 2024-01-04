import copy
import dataclasses
import numpy as np
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, TypeVar
from ..common import Model, ModelParameter
from ..utils import Array


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float
    num_spectrum_samples = float

TModel = TypeVar("TModel", bound=Type[Model])


def param_like(
    template_param: ModelParameter, overrides: Optional[Dict[str, Any]] = None
) -> ModelParameter:
    """Returns a new parameter based on a template.

    :param template_param: the returned parameter is a (deep) copy of the template
      parameter.
    :param overrides: optional dictionary of attributes of the template parameter to
      replace.
    """
    new_param = copy.deepcopy(template_param)

    overrides = overrides or {}
    for attr, value in overrides.items():
        setattr(new_param, attr, value)

    return new_param


@dataclasses.dataclass
class PeriodicModelParameter(ModelParameter):
    period: float = 1
    offset: float = 0
    lower_bound: float = dataclasses.field(init=False)
    upper_bound: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.lower_bound = 1.5 * self.offset
        self.upper_bound = 1.5 * (self.offset + self.period)

    upper_bound = 1.5 * np.pi

    def clip(self, value: float):
        """Clip value to lie between lower and upper bounds."""
        value = value - self.offset
        return (value % self.period) + self.offset


def rescale_model_x(model_class: TModel, x_scale: float) -> TModel:
    """Rescales the x-axis for a model class.

    This is commonly used to convert models between linear and angular units.

    :param model_class: model class to rescale
    :param x_scale: multiplicative x-axis scale factor. To convert a model that takes
      x in angular units and convert to one that takes x in linear units use
      `x_scale = 2 * np.pi`
    """

    class ScaledModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__x_scale = x_scale
            self.__rescale = True

        def func(
            self, x: Array[("num_samples",), np.float64], param_values: Dict[str, float]
        ) -> Array[("num_samples", "num_y_channels"), np.float64]:
            x = (x * self.__x_scale) if self.__rescale else x
            return super().func(x, param_values)

        def estimate_parameters(
            self,
            x: Array[("num_samples",), np.float64],
            y: Array[("num_samples", "num_y_channels"), np.float64],
        ):
            # avoid double rescaling if estimate_parameters calls self.func internally
            self.__rescale = False
            super().estimate_parameters(x * self.__x_scale, y)
            self.__rescale = True

    ScaledModel.__name__ = model_class.__name__
    ScaledModel.__doc__ = model_class.__doc__

    return ScaledModel
