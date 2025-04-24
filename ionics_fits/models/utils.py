import copy
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ..common import TSCALE_FUN, ModelParameter

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
    r"""Represents a model parameter whose value is periodic.

    Parameter values are clipped to lie within::

        ((value - offset) % period) + offset

    ``PeriodicModelParameter``\s do not support bounds.

    Attributes:
        period: the period (default = 1)
        offset: the offset (default = 0)
    """

    scale_func: TSCALE_FUN
    period: float = 1
    offset: float = 0
    lower_bound: float = dataclasses.field(init=False, default=-np.inf)
    upper_bound: float = dataclasses.field(init=False, default=+np.inf)

    def __post_init__(self):
        super().__post_init__()
        self._metadata_attrs += ["period", "offset"]

    def __setattr__(self, name, value):
        if name in ["lower_bound", "upper_bound"]:
            raise ValueError("PeriodicModelParameter does not support bounds")
        super().__setattr__(name, value)

    def _format_metadata(self) -> List[str]:
        metadata = super()._format_metadata()

        metadata = [
            attr
            for attr in metadata
            if not (attr.startswith("lower_bound=") or attr.startswith("upper_bound="))
        ]

        metadata.append(f"period={self.period:.3f}")
        metadata.append(f"offset={self.offset:.3f}")

        return metadata

    def __repr__(self):
        return super().__repr__()

    def clip(self, value: float):
        """Clip value to lie between lower and upper bounds."""
        return ((value - self.offset) % self.period) + self.offset
