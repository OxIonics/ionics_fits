from typing import TYPE_CHECKING

from .common import Model2D
from .. import models


if TYPE_CHECKING:
    num_samples_ax_0 = float
    num_samples_ax_1 = float
    num_y_channels = float


class Gaussian2D(Model2D):
    """2D Gaussian. See ionics_fits.models.Gaussian for details.

    Document attributes. (As Gaussian with _x / _y for the two axes. Single, shared
    amplitude).

    What do we do with y0? Should be forced to 0?
    """

    def __init__(self):
        outer_model = models.Gaussian()
        outer_model.parameters["y0"].fixed_to = 0

        super().__init__(
            models=[models.Gaussian(), outer_model],
            result_params=["a"],
            param_renames={"a_y": "a", "y0_x": "y0", "y0_x": "y0"},
        )


class Cone(Model2D):
    """2D Cone Model. See ionics_fits.models.ConeSegment for details"""

    def __init__(self):
        super().__init__(
            models=[models.ConeSegment(), models.Triangle()],
            result_params=["gamma"],
            param_renames=None,
        )
