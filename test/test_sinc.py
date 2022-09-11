import random
from typing import Dict, Optional, Tuple
import numpy as np
import test

import ionics_fits as fits


def test_sinc():
    """Test for sinc.Sinc"""
    x = np.linspace(-10, 10, 100)
    params = {
        "x0": [-3, 0, 1, 5],
        "y0": [-1, 0, 10],
        "a": [-1, 4],
        "w": [1, 3, 10],
    }
    model = fits.models.Sinc()
    test.common.check_multiple_param_sets(
        x,
        model,
        params,
        test.common.TestConfig(plot_failures=True),
    )


def test_sinc2():
    """Test for sinc.Sinc2"""
    x = np.linspace(-10, 10, 100)
    params = {
        "x0": [-3, 0, 1, 5],
        "y0": [-1, 0, 10],
        "a": [-1, 4],
        "w": [1, 3, 10],
    }
    model = fits.models.Sinc2()
    test.common.check_multiple_param_sets(
        x,
        model,
        params,
        test.common.TestConfig(plot_failures=True),
    )


def sinc_param_generator(
    fuzzed_params: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    values = test.common.generate_param_set(fuzzed_params)
    if random.choice([True, False]):
        values["a"] = -values["a"]
    return values


def sinc_fuzzer(
    model,
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(-10, 10, 100)
    fuzzed_params = {
        "x0": [-5, 5],
        "y0": [-10, 10],
        "a": [0.25, 4],
        "w": [1, 10],
    }
    static_params = {}
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
        param_generator=sinc_param_generator,
    )


def fuzz_sinc(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    sinc_fuzzer(
        model=fits.models.sinc.Sinc(),
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )


def fuzz_sinc2(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    sinc_fuzzer(
        model=fits.models.sinc.Sinc2(),
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )
