import numpy as np

from ionics_fits.models.thermometry import SidebandHeatingRate
from .common import check_multiple_param_sets, Config


def test_sideband_heating(plot_failures):
    """Test for thermometry.SidebandHeatingRate"""
    t_heating = np.linspace(0, 1, 15)

    params = {
        "n_bar_0": 0.1,
        "n_bar_dot": 2,
    }
    model = SidebandHeatingRate()
    check_multiple_param_sets(
        t_heating,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=3e-2),
    )
