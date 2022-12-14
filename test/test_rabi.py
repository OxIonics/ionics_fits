from typing import Optional
import numpy as np

import ionics_fits as fits
from . import common


def test_rabi_freq():
    """Test for rabi.RabiFlopFreq"""
    w = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    t_pulse = 5e-6
    params = {
        "P_readout_e": 1.0,
        "P_readout_g": 0.0,
        "w_0": 0.5e6 * 2 * np.pi,
        "omega": np.array([0.1, 0.25, 1]) * np.pi / t_pulse,
        "t_pulse": t_pulse,
        "t_dead": 0.0,
        "tau": np.inf,
    }

    model = fits.models.RabiFlopFreq(start_excited=True)
    common.check_multiple_param_sets(
        w,
        model,
        params,
        common.TestConfig(plot_failures=True),
    )


def test_rabi_time():
    """Test for rabi.RabiFlopTime"""
    t_pulse = np.linspace(0, 20e-6, 100) * 2 * np.pi
    params = {
        "P_readout_e": 1.0,
        "P_readout_g": 0.0,
        "w_0": 2 * np.pi * np.array([0, 0.025e6]),
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
        "w": 0.0,
        "t_dead": 0.0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopTime(start_excited=True)
    common.check_multiple_param_sets(
        t_pulse,
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
        "delta": 2 * np.pi * np.array([-0.5e6, 0.5e6]),
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
        "t_pulse": (3e-6, 5e-6),
    }

    static_params = {
        "P_readout_e": 1.0,
        "P_readout_g": 0.0,
        "t_dead": 0.0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopFreq(start_excited=True)
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
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
    }

    static_params = {
        "P_readout_e": 1.0,
        "P_readout_g": 0.0,
        "t_dead": 0.0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopTime(start_excited=True)
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
