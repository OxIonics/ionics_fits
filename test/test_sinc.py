import random
from typing import Dict, Optional, Tuple

import numpy as np

from ionics_fits.models.sinc import Sinc, Sinc2

from .common import (
    Config,
    check_multiple_param_sets,
    check_single_param_set,
    fuzz,
    generate_param_set,
)


def test_sinc(plot_failures: bool):
    """Test for sinc.Sinc"""
    x = np.linspace(-20, 20, 200)
    params = {
        "x0": [-13, -1, 0, 1, 5],
        "y0": [-1, 0, 10],
        "a": [-1, 4],
        "w": [1, 3, 10],
    }
    model = Sinc()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.25),
    )


def test_sinc_heuristic(plot_failures: bool):
    """Test sinc.Sinc heuristic gives an accurate estimate in an easy case"""
    x = np.linspace(-20, 20, 200)
    params = {
        "x0": 5,
        "y0": 2,
        "a": -4,
        "w": 3,
    }
    model = Sinc()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.05),
    )


def test_sinc2(plot_failures: bool):
    """Test for sinc.Sinc2"""
    x = np.linspace(-20, 20, 400)
    params = {
        "x0": [-8, -1, 0, 1, 5, 15],
        "y0": [-1, 0, 10],
        "a": [-1, 4],
        "w": [1, 3, 10],
    }
    model = Sinc2()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.1),
    )


def test_sinc2_heuristic(plot_failures: bool):
    """Test sinc.Sinc2 heuristic gives an accurate estimate in an easy case"""
    x = np.linspace(-20, 20, 200)
    params = {
        "x0": 5,
        "y0": 2,
        "a": -4,
        "w": 3,
    }
    model = Sinc2()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.02),
    )


def sinc_param_generator(
    fuzzed_params: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    values = generate_param_set(fuzzed_params)
    if random.choice([True, False]):
        values["a"] = -values["a"]
    return values


def sinc_fuzzer(
    model,
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[Config] = None,
) -> float:
    x = np.linspace(-10, 20, 500)
    fuzzed_params = {
        "x0": (-8, 15),
        "y0": (-10, 10),
        "a": (0.25, 4),
        "w": (1, 10),
    }

    return fuzz(
        x=x,
        model=model,
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=sinc_param_generator,
    )


def fuzz_sinc(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    return sinc_fuzzer(
        model=Sinc(),
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )


def fuzz_sinc2(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    return sinc_fuzzer(
        model=Sinc2(),
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )
