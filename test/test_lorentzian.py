from typing import Optional
import numpy as np
import test

import ionics_fits as fits


def test_lorentzian():
    """Test for lorentzian.Lorentzian"""
    x = np.linspace(-4, 4, 1000)
    params = {
        "x0": [-1, 0, 0.25, 2],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "fwhmh": [0.1, 0.25, 1],
    }
    model = fits.models.Lorentzian()
    test.common.check_multiple_param_sets(
        x,
        model,
        params,
        test.common.TestConfig(plot_failures=True),
    )


def fuzz_lorentzian(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(-4, 4, 1000)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 1),
        "a": (-5, 5),
        "fwhmh": (0.1, 1),
    }
    static_params = {}

    model = fits.models.Lorentzian()
    test_config = test_config or test.common.TestConfig()
    test_config.plot_failures = True

    return test.common.fuzz(
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
