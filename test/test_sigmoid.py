import numpy as np

from ionics_fits.models.sigmoid import LogisticFunction
from .common import check_multiple_param_sets, Config


def test_logistic_function(plot_failures):
    """Test for sigmoid.LogisticFunction"""
    x = np.linspace(-10, 20, 500)
    params = {
        "x0": [-3, 1],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "k": [1],
    }
    model = LogisticFunction()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.1),
    )
