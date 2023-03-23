import numpy as np

import ionics_fits as fits
from . import common


def test_lorentzian(plot_failures):
    """Test for lorentzian.Lorentzian"""
    x = np.linspace(-4, 4, 1000)
    params = {
        "x0": [-1, 0, 0.25, 2],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "fwhmh": [0.1, 0.25, 1],
    }
    model = fits.models.Lorentzian()
    common.check_multiple_param_sets(
        x,
        model,
        params,
        common.TestConfig(plot_failures=plot_failures),
    )


def fuzz_lorentzian(
    num_trials: int,
    stop_at_failure: bool,
    test_config: common.TestConfig,
) -> float:
    x = np.linspace(-4, 4, 1000)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 1),
        "a": (-5, 5),
        "fwhmh": (0.1, 1),
    }

    return common.fuzz(
        x=x,
        model=fits.models.Lorentzian(),
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
