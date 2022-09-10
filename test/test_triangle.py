from typing import Dict, Optional, Tuple
import numpy as np
import test

import ionics_fits as fits


def test_triangle():
    """Test for triangle.Triangle"""
    x = np.linspace(-2, 2, 100)
    params = {
        "x0": [-1, +1],
        "y0": [-1, +1],
        "k": [-5, +5],
        "dk": [0, 1, -1],
        "y_min": -np.inf,
        "y_max": +np.inf,
    }
    model = fits.models.Triangle()
    test.common.check_multiple_param_sets(
        x,
        model,
        params,
        test.common.TestConfig(plot_failures=True),
    )


def triangle_param_generator(
    fuzzed_params: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    fuzzed_param_values = {
        param: np.random.uniform(*fuzzed_params[param]) for param in ["x0", "y0", "k"]
    }
    k = fuzzed_param_values["k"]
    fuzzed_param_values["dk"] = np.random.uniform(-0.5 * k, +0.5 * k)

    return fuzzed_param_values


def fuzz_triangle(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(-2, 2, 100)
    fuzzed_params = {
        "x0": [-1, +1],
        "y0": [-1, +1],
        "k": [-5, +5],
        "dk": [-0.5, 0.5],
    }
    static_params = {"y_min": -np.inf, "y_max": +np.inf}

    model = fits.models.Triangle()
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
        param_generator=triangle_param_generator,
    )
