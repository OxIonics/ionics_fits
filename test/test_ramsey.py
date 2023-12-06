import numpy as np

import ionics_fits as fits
from . import common


def test_ramsey_w0_phi_only(plot_failures: bool):
    """Test for ramsey.Ramsey in the case where w_0 and phi are the only unknowns.

    This case is special-cased in the parameter estimator.
    """
    w = np.linspace(-2e6, 2e6, 500) * 2 * np.pi
    t_ramsey = 5e-6
    t_pi_2 = t_ramsey / 10
    params = {
        "P_readout_e": 1,
        "P_readout_g": 0,
        "t": t_ramsey,
        "t_pi_2": t_pi_2,
        "phi": np.array([0.0, 0.1, 0.25, 0.5, 1]) * 2 * np.pi,
        "w_0": [0, 0.1, 1, 3, 5],
        "tau": np.inf,
    }

    model = fits.models.Ramsey(start_excited=True)
    model.parameters["P_readout_e"].fixed_to = params["P_readout_e"]
    model.parameters["P_readout_g"].fixed_to = params["P_readout_g"]
    model.parameters["t"].fixed_to = params["t"]
    model.parameters["t_pi_2"].fixed_to = params["t_pi_2"]

    common.check_multiple_param_sets(
        w,
        model,
        params,
        common.TestConfig(
            plot_failures=plot_failures, param_tol=None, residual_tol=1e-4
        ),
    )
