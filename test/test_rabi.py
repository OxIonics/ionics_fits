from typing import Optional
import numpy as np
import test

import ionics_fits as fits


def test_rabi_freq():
    """Test for rabi.RabiFlopFreq"""
    delta = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "detuning_offset": 0.5e6 * 2 * np.pi,
        "omega": [0.25 * np.pi / 5e-6, np.pi / 5e-6],
        "t_pulse": 5e-6,
        "t_dead": 0,
        "tau": np.inf,
    }

    model = fits.models.RabiFlopFreq()
    test.common.check_multiple_param_sets(
        delta,
        model,
        params,
        test.common.TestConfig(plot_failures=True),
    )


def fuzz_rabi_freq(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    delta = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    fuzzed_params = {
        "detuning_offset": [-0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi],
        "omega": [0.25 * np.pi / 5e-6, np.pi / 5e-6],
        "t_pulse": [3e-6, 5e-6],
    }

    static_params = {
        "P1": 1,
        "P_upper": 1,
        "P_lower": 0,
        "t_dead": 0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopFreq()
    test_config = test_config or test.common.TestConfig()
    test_config.plot_failures = True

    return test.common.fuzz(
        x=delta,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
