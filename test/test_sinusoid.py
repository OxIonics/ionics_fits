import numpy as np

import ionics_fits as fits
from . import common


def test_sinusoid(plot_failures: bool):
    """Test for sinusoid.Sinusoid"""
    x = np.linspace(-10, 10, 1000)
    params = {
        "a": 2,
        "omega": [1 / (2 * np.pi), 5 / (2 * np.pi), 10 / (2 * np.pi)],
        "phi": 0.5,
        "y0": 1,
        "x0": 0,
        "tau": np.inf,
        "P_lower": 0,
        "P_upper": 0,
    }
    model = fits.models.Sinusoid()
    common.check_multiple_param_sets(
        x, model, params, common.TestConfig(plot_failures=plot_failures)
    )


def test_sinusoid_x0(plot_failures: bool):
    """Test for sinusoid.Sinusoid with `x0` floated instead of `phi`"""
    x = np.linspace(-5, 20, 1000)
    params = {
        "a": 2,
        "omega": 1 / (2 * np.pi),
        "phi": 0,
        "y0": 1,
        "x0": 1,
        "tau": np.inf,
        "P_lower": 0,
        "P_upper": 0,
    }
    model = fits.models.Sinusoid()
    model.parameters["x0"].fixed_to = None
    model.parameters["phi"].fixed_to = 0
    common.check_single_param_set(
        x, model, params, common.TestConfig(plot_failures=plot_failures)
    )


def fuzz_sinusoid(
    num_trials: int,
    stop_at_failure: bool,
    test_config: common.TestConfig,
) -> float:
    x = np.linspace(-2, 4, 1000)

    dx = x.ptp() / x.size
    w_nyquist = 0.5 * (1 / dx) * 2 * np.pi
    w_min = 1 / x.ptp() * 2 * np.pi

    static_params = {
        "tau": np.inf,
        "x0": 0,
        "P_lower": 0,
        "P_upper": 0,
    }
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
    return common.fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
