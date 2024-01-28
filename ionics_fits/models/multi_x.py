from typing import TYPE_CHECKING

from .containers import MappedModel, Model2D
from .cone import ConeSlice
from .gaussian import Gaussian
from .polynomial import Parabola
from .triangle import Triangle


if TYPE_CHECKING:
    num_samples_ax_0 = float
    num_samples_ax_1 = float
    num_y_channels = float


class Gaussian2D(Model2D):
    """2D Gaussian according to:
    ```
    z = (
        a / ((sigma * sqrt(2*pi)) * (sigma * sqrt(2*pi)))
        * exp(-0.5*((x-x0)/(sigma_x))^2 -0.5*((y-y0)/(sigma_y))^2) + z0
    ```
    Parameters are:
      - a
      - x0
      - y0
      - sigma_x
      - sigma_y
      - z0

    Derived results are:
      - FWHMH_x
      - FWHMH_y
      - w0_x
      - w0_y
      - peak_y

    See ionics_fits.models.Gaussian for details.
    """

    def __init__(self):
        # TODO: once we have proper support for 2D models, we should provide
        # non-empty model names and Wrap the 2D model rather than wrapping the 1D
        # models
        inner_model = MappedModel(
            model=Gaussian(),
            param_mapping={"a_x": "a", "x0_x": "x0", "sigma_x": "sigma", "z0": "y0"},
            derived_result_mapping={"FWHMH_x": "FWHMH", "w0_x": "w0", None: "peak"},
        )
        outer_model = MappedModel(
            model=Gaussian(),
            param_mapping={
                "a": "a",
                "x0_y": "x0",
                "sigma_y": "sigma",
            },
            fixed_params={"y0": 0},
            derived_result_mapping={"FWHMH_y": "FWHMH", "w0_y": "w0"},
        )

        super().__init__(
            models=(inner_model, outer_model),
            model_names=("", ""),
            result_params=("a_x",),
        )


class Parabola2D(Model2D):
    """2D Parabola according to:
    ```
    z = k_x * (x - x0)^2 + k_y *(y - y0) + z0
    ```

    Parameters are:
      - x0
      - y0
      - k_x
      - k_y
      - y0

    See ionics_fits.models.Parabola for details.
    """

    def __init__(self):
        # TODO: once we have proper support for 2D models, we should provide
        # non-empty model names and Wrap the 2D model rather than wrapping the 1D
        # models
        inner_model = MappedModel(
            model=Parabola(),
            param_mapping={"x0": "x0", "y0_x": "y0", "k_x": "k"},
        )
        outer_model = MappedModel(
            model=Parabola(),
            param_mapping={"y0": "x0", "z0": "y0", "k_y": "k"},
        )

        super().__init__(
            models=(inner_model, outer_model),
            model_names=("", ""),
            result_params=("y0_x",),
        )


class Cone2D(Model2D):
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
        # TODO: once we have proper support for 2D models, we should provide
        # non-empty model names and Wrap the 2D model rather than wrapping the 1D
        # models
        inner_model = MappedModel(
            model=ConeSlice(),
            param_mapping={"alpha": "alpha", "y0": "z0", "x0_x": "x0", "k_x": "k"},
        )

        triangle = Triangle()
        triangle.parameters["y0"].fixed_to = 0
        outer_model = MappedModel(
            model=triangle,
            param_mapping={"x0_y": "x0", "k_y": "k"},
            fixed_params={
                param_name: param_data.fixed_to
                for param_name, param_data in triangle.parameters.items()
                if param_data.fixed_to is not None
            },
        )

        super().__init__(
            models=(inner_model, outer_model),
            model_names=("", ""),
            result_params=("alpha",),
        )
