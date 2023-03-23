import numpy as np

import ionics_fits as fits
from . import common


def test_triangle(plot_failures: bool):
    """Test for triangle.Triangle"""
    x = np.linspace(-2, 2, 100)
    params = {
        "x0": [-1, +1],
        "y0": [-1, +1],
        "k": [-5, +5],
        "sym": [0, 0.5, -0.5],
        "y_min": -np.inf,
        "y_max": +np.inf,
    }
    model = fits.models.Triangle()
    model.parameters["sym"].fixed_to = None

    common.check_multiple_param_sets(
        x,
        model,
        params,
        common.TestConfig(plot_failures=plot_failures),
    )


def fuzz_triangle(
    num_trials: int,
    stop_at_failure: bool,
    test_config: common.TestConfig,
) -> float:
    x = np.linspace(-2, 2, 100)
    fuzzed_params = {
        "x0": (-1, +1),
        "y0": (-1, +1),
        "k": (-5, +5),
        "sym": (-0.5, 0.5),
    }
    static_params = {"y_min": -np.inf, "y_max": +np.inf}

    model = fits.models.Triangle()
    model.parameters["sym"].fixed_to = None

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
