import numpy as np

import ionics_fits as fits
from . import common


def test_gaussian(plot_failures):
    """Test for gaussian.Gaussian"""
    x = np.linspace(-10, 20, 500)
    params = {
        "x0": [-3, 1, 0, 0.25, 10],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "sigma": [0.1, 0.25, 1],
    }
    model = fits.models.Gaussian()
    common.check_multiple_param_sets(
        x,
        model,
        params,
        common.TestConfig(plot_failures=plot_failures, heuristic_tol=0.2),
    )


def fuzz_gaussian(
    num_trials: int,
    stop_at_failure: bool,
    test_config: common.TestConfig,
) -> float:
    x = np.linspace(-10, 10, 500)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 0, 1),
        "a": (-5, 5),
        "sigma": (0.1, 0.25, 1),
    }

    return common.fuzz(
        x=x,
        model=fits.models.Gaussian(),
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
