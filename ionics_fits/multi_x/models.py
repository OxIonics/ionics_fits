from typing import TYPE_CHECKING

from .common import Model2D
from .. import models


if TYPE_CHECKING:
    num_samples_ax_0 = float
    num_samples_ax_1 = float
    num_y_channels = float


class Gaussian2D(Model2D):
    """2D Gaussian according to:
    ```
    y = (
        a / ((sigma * sqrt(2*pi)) * (sigma * sqrt(2*pi)))
        * exp(-0.5*((x-x0)/(sigma_x))^2 -0.5*((y-y0)/(sigma_y))^2) + y0
    ```
    Parameters are:
      - a
      - x0
      - y0
      - sigma_x
      - sigma_y
      - y0

    Derived results are:
      - FWHMH_x
      - FWHMH_y
      - w0_x
      - w0_y
      - peak_y

    See ionics_fits.models.Gaussian for details.
    """

    def __init__(self):
        outer_model = models.Gaussian()
        outer_model.parameters["y0"].fixed_to = 0

        super().__init__(
            models=(models.Gaussian(), outer_model),
            model_names=("x", "y"),
            result_params=("a",),
            param_renames={"a_y": "a", "y0_x": "y0", "y0_x": "y0", "y0_y": None},
        )


class Cone(Model2D):
    """2D Cone Model.

    Parameters are (x params inherit from :class :class ionics_fits.models.ConeSlide:
    while y params inherit from :class ionics_fits.models.Triangle:
      - x0_x
      - x0_y
      - k_x
      - k_y
      - y0

     See ionics_fits.models.ConeSlice and ionics_fits.models.triangle for details.
    """

    def __init__(self):
        triangle = models.Triangle()
        # triangle.parameters["y0"].fixed_to = 0

        super().__init__(
            models=(models.ConeSlice(), triangle),
            model_names=("x", "y"),
            result_params=("alpha",),
            param_renames={
                "y0_y": None,
                "z0_x": "y0",
                "sym_y": None,
                "y_min_y": None,
                "y_max_y": None,
                "k_p_y": None,
                "k_m_y": None,
            },
        )
