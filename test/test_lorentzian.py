import numpy as np

from ionics_fits.models.lorentzian import Lorentzian
from .common import check_multiple_param_sets, check_single_param_set, fuzz, Config


def test_lorentzian(plot_failures):
    """Test for lorentzian.Lorentzian"""
    x = np.linspace(-4, 4, 1000)
    params = {
        "x0": [-1, 0, 0.25, 2],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "fwhmh": [0.1, 0.25, 0.5],
    }
    model = Lorentzian()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.7),
    )


def test_lorentzian_heuristic(plot_failures):
    """Test lorentzian.Lorentzian heuristic is accurate in an easy situation"""
    x = np.linspace(-4, 4, 1000)
    params = {
        "x0": 0.25,
        "y0": 1,
        "a": -5,
        "fwhmh": 0.25,
    }
    model = Lorentzian()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.05),
    )


def fuzz_lorentzian(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-4, 4, 1000)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 1),
        "a": (-5, 5),
        "fwhmh": (0.1, 1),
    }

    return fuzz(
        x=x,
        model=Lorentzian(),
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
