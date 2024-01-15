import numpy as np

import ionics_fits as fits
from . import common


def test_scaled_model(plot_failures):
    """Test for ScaledModel"""
    x = np.linspace(-10, 10, 200)
    params = {"a": 2.5, "omega": 3.3, "phi": 1, "y0": -6, "tau": np.inf, "x0": 0}

    model = fits.models.Sinusoid()
    scaled_model = fits.models.ScaledModel(model=model, x_scale=2)

    common.is_close(model(2 * x, **params), scaled_model(x, **params), 1e-9)

    common.check_single_param_set(
        x=x,
        model=scaled_model,
        test_params=params,
        config=common.TestConfig(plot_failures=plot_failures, heuristic_tol=0.1),
    )
