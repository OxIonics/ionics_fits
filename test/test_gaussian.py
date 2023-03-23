from typing import Optional
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
        common.TestConfig(plot_failures=plot_failures),
    )


def fuzz_gaussian(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    x = np.linspace(-10, 10, 500)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 0, 1),
        "a": (-5, 5),
        "sigma": (0.1, 0.25, 1),
    }
    static_params = {}

    model = fits.models.Gaussian()
    test_config = test_config or common.TestConfig()
    test_config.plot_failures = True

    return common.fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
