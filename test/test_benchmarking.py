import numpy as np

from ionics_fits.models.benchmarking import Benchmarking

from .common import Config, check_multiple_param_sets, fuzz


def test_benchmarking(plot_failures):
    """Test for benchmarking.Benchmarking"""
    x = np.linspace(1, 1000, 300)
    params = {"p": [0.1, 0.3, 0.9], "y0": [0.9, 0.99], "y_inf": 1 / 2**2}
    model = Benchmarking(num_qubits=2)
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )


def fuzz_benchmarking(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(1, 1000, 300)
    fuzzed_params = {
        "p": (0.1, 0.3),
        "y0": (0.9, 0.99),
    }
    static_params = {"y_inf": 1 / 2**2}

    return fuzz(
        x=x,
        model=Benchmarking(num_qubits=2),
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
