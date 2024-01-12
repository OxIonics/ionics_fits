import numpy as np

import ionics_fits as fits
from . import common


def test_logistic_function(plot_failures):
    """Test for sigmoid.LogisticFunction"""
    x = np.linspace(-10, 20, 500)
    params = {
        "x0": [-3, 1],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "k": [1],
    }
    model = fits.models.LogisticFunction()
    common.check_multiple_param_sets(
        x,
        model,
        params,
        common.TestConfig(plot_failures=plot_failures, heuristic_tol=0.1),
    )
