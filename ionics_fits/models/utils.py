import copy
import dataclasses
import numpy as np
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, TypeVar

from ..common import Model, ModelParameter


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
