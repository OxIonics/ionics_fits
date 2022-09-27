from typing import Optional
import numpy as np

import ionics_fits as fits
from . import common


def test_rabi_freq():
    """Test for rabi.RabiFlopFreq"""
    delta = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "delta": 0.5e6 * 2 * np.pi,
        "omega": np.array([0.1, 0.25, 1]) * np.pi / 5e-6,
        "t_pulse": 5e-6,
        "t_dead": 0,
        "tau": np.inf,
    }

    model = fits.models.RabiFlopFreq()
    common.check_multiple_param_sets(
        delta,
        model,
        params,
        common.TestConfig(plot_failures=True),
    )


def test_rabi_time():
    """Test for rabi.RabiFlopTime"""
    t = np.linspace(0, 20e-6, 100) * 2 * np.pi
    params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "delta": [0, 0.025e6 * 2 * np.pi],
        "omega": [0.25 * np.pi / 5e-6, np.pi / 5e-6],
        "t_dead": 0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopTime()
    common.check_multiple_param_sets(
        t,
        model,
        params,
        common.TestConfig(plot_failures=True, param_tol=None, residual_tol=1e-4),
    )


def fuzz_rabi_freq(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    delta = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    fuzzed_params = {
        "delta": (-0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi),
        "omega": (0.25 * np.pi / 5e-6, np.pi / 5e-6),
        "t_pulse": (3e-6, 5e-6),
    }

    static_params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "t_dead": 0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopFreq()
    test_config = test_config or common.TestConfig()
    test_config.plot_failures = True

    return common.fuzz(
        x=delta,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )


def fuzz_rabi_time(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    t = np.linspace(0, 20e-6, 400) * 2 * np.pi
    fuzzed_params = {
        "delta": (0, 0.3e6 * 2 * np.pi),
        "omega": (0.25 * np.pi / 5e-6, np.pi / 5e-6),
    }

    static_params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "t_dead": 0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopTime()
    test_config = test_config or common.TestConfig()
    test_config.plot_failures = True

    return common.fuzz(
        x=t,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
