import numpy as np

from ionics_fits.models.sinusoid import Sinusoid
from ionics_fits.models.transformations.scaled_model import ScaledModel

from .common import is_close, check_single_param_set, Config


def test_scaled_model(plot_failures):
    """Test for ScaledModel"""
    x = np.linspace(-3, 10, 500)
    params = {"a": 2.5, "omega": 3.3, "phi": 1, "y0": -6, "tau": np.inf, "x0": 0}

    model = Sinusoid()
    scaled_model = ScaledModel(model=model, x_scale=4)

    assert is_close(model(4 * x, **params), scaled_model(x, **params), 1e-9)

    check_single_param_set(
        x=x,
        model=scaled_model,
        test_params=params,
        config=Config(plot_failures=plot_failures, heuristic_tol=0.2),
    )
