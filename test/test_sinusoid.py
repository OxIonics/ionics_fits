import numpy as np
import test
from typing import Optional

import ionics_fits as fits


def test_sinusoid():
    """Test for sinusoid.Sinusoid"""
    x = np.linspace(-5, 20, 1000) * 2 * np.pi
    params = {
        "a": 2,
        "omega": 1,
        "phi": 0.5,
        "y0": 1,
        "x0": 0,
        "tau": np.inf,
    }
    model = fits.models.Sinusoid()
    test.common.check_single_param_set(
        x, model, params, test.common.TestConfig(plot_failures=True)
    )


def test_sinusoid_x0():
    """Test for sinusoid.Sinusoid with `x0` floated instead of `phi`"""
    x = np.linspace(-5, 20, 1000) * 2 * np.pi
    params = {
        "a": 2,
        "omega": 1,
        "phi": 0,
        "y0": 1,
        "x0": 1,
        "tau": np.inf,
    }
    model = fits.models.Sinusoid()
    model.parameters["x0"].fixed_to = None
    model.parameters["phi"].fixed_to = 0
    test.common.check_single_param_set(x, model, params)


def fuzz_sinusoid(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(-2, 4, 1000) * 2 * np.pi

    dx = x.ptp() / x.size
    w_nyquist = 0.5 * (1 / dx) * 2 * np.pi
    w_min = 1 / x.ptp() * 2 * np.pi

    static_params = {"tau": np.inf, "x0": 0}
    fuzzed_params = {
        "a": (0, 3),
        "omega": (2 * w_min, 0.5 * w_nyquist),
        "phi": (-np.pi, np.pi),
        "y0": (-3, 3),
    }
    # don't fail if a phase of ~ pi fits to ~ -pi
    test_config.param_tol = None
    test_config.residual_tol = 1e-3

    model = fits.models.Sinusoid()
    return test.common.fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
