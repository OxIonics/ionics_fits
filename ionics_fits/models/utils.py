import copy
import dataclasses
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..common import ModelParameter, TSCALE_FUN


if TYPE_CHECKING:
    num_samples = float
    num_spectrum_samples = float


def param_like(
    template_param: ModelParameter, overrides: Optional[Dict[str, Any]] = None
) -> ModelParameter:
    """Returns a new parameter based on a template.

    :param template_param: the returned parameter is a (deep) copy of the template
      parameter.
    :param overrides: optional dictionary of attributes of the template parameter to
      replace.
    :returns: the new :class:`~ionics_fits.common.ModelParameter`
    """
    new_param = copy.deepcopy(template_param)

    overrides = overrides or {}
    for attr, value in overrides.items():
        setattr(new_param, attr, value)

    return new_param


@dataclasses.dataclass
class PeriodicModelParameter(ModelParameter):
    """ Represents a model parameter whose value is periodic.
    
    Parameter values are clipped to lie within::

        value = value - offset
        value (value % period) + offset

    Attributes:
        period: the period (default = 1)
        offset: the offset (default = 0)
    """
    scale_func: TSCALE_FUN
    period: float = 1
    offset: float = 0
    lower_bound: float = dataclasses.field(init=False)
    upper_bound: float = dataclasses.field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._metadata_attrs += ["period", "offset"]

        self.lower_bound = 1.5 * self.offset
        self.upper_bound = 1.5 * (self.offset + self.period)

    def _format_metadata(self) -> List[str]:
        metadata = super()._format_metadata()
        
        metadata = [
            attr for attr in metadata
            if not (attr.startswith("lower_bound=") or attr.startswith("upper_bound="))
        ]

        metadata.append(f"period={self.period:.3f}")
        metadata.append(f"offset={self.offset:.3f}")

        return metadata

    def __repr__(self):
        return super().__repr__()

    def clip(self, value: float):
        """Clip value to lie between lower and upper bounds."""
        value = value - self.offset
        return (value % self.period) + self.offset
