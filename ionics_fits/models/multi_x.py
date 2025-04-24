"""
Models with more than one x-axis degree of freedom, which have been created from 1D
models using :class:`~ionics_fits.models.transformations.model_2d.Model2D`.
"""

import numpy as np

from ..utils import TX_SCALE, TY_SCALE, to_float
from .cone import ConeSlice
from .gaussian import Gaussian
from .polynomial import Parabola
from .transformations.mapped_model import MappedModel
from .transformations.model_2d import Model2D
from .triangle import Triangle


class Gaussian2D(MappedModel):
    """2D Gaussian according to::

        y = (
            a / ((sigma_x0 * sqrt(2*pi)) * (sigma_x1 * sqrt(2*pi)))
            * exp(-0.5*((x0-x0_x0)/(sigma_x0))^2 -0.5*((x1-x0_x1)/(sigma_x1))^2) + y0

    Parameters are:

    * ``a``
    * ``x0_x0``
    * ``x0_x1``
    * ``sigma_x0``
    * ``sigma_x1``
    * ``y0``

    Derived results are:
      - ``FWHMH_x0``
      - ``FWHMH_x1``
      - ``w0_x0``
      - ``w0_x1``
      - ``peak``

    See :class:`~ionics_fits.models.gaussian.Gaussian` for details.
    """

    def __init__(self):
        class _Gaussian2D(Model2D):
            def __init__(self):
                super().__init__(
                    models=(Gaussian(), Gaussian()),
                    result_params=("a",),
                )

            def wrap_scale_funcs(self):
                super().wrap_scale_funcs()

                def scale_power(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
                    return np.prod(x_scales) * to_float(y_scales)

                self.parameters["a_x1"].scale_func = scale_power

        model = _Gaussian2D()

        param_mapping = {
            param_name: param_name
            for param_name in model.parameters.keys()
            if param_name not in ["y0_x0", "y0_x1", "a_x1"]
        }
        param_mapping["y0"] = "y0_x0"
        param_mapping["a"] = "a_x1"
        super().__init__(
            model=model,
            param_mapping=param_mapping,
            fixed_params={"y0_x1": 0},
            derived_result_mapping={"peak": "peak_x1", None: "peak_x0"},
        )


class Parabola2D(MappedModel):
    """2D Parabola according to::

        y = k_x0 * (x0 - x0_x0)^2 + k_x1 *(x1 - x0_x1) + y0


    Parameters are:

    * ``x0_x0``
    * ``x0_x1``
    * ``k_x0``
    * ``k_x1``
    * ``y0``

    See :class:`~ionics_fits.models.polynomial.Parabola` for details.
    """

    def __init__(self):
        model = Model2D(models=(Parabola(), Parabola()), result_params=("y0",))
        param_mapping = {
            param_name: param_name
            for param_name in model.parameters.keys()
            if param_name != "y0_x1"
        }
        param_mapping["y0"] = "y0_x1"
        super().__init__(model=model, param_mapping=param_mapping)


class Cone2D(MappedModel):
    r"""2D Cone Model.

    Parameters are:

    - ``x0_x0``
    - ``x0_x1``
    - ``k_x0``
    - ``k_x1``
    - ``y0``

    Parameters with an ``_x0`` suffix inherit from
    :class:`~ionics_fits.models.cone.ConeSlice`\, parameters with an ``_x1`` suffix
    inherit from :class:`~ionics_fits.models.triangle.Triangle`.
    """

    def __init__(self):
        model = Model2D(models=(ConeSlice(), Triangle()), result_params=("alpha",))
        param_mapping = {
            param_name: param_name
            for param_name in model.parameters.keys()
            if param_name != "z0_x0"
        }

        param_mapping["y0"] = "z0_x0"

        super().__init__(model=model, param_mapping=param_mapping)
