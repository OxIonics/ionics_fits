import numpy as np

from ionics_fits.models.exponential import Exponential

from .common import Config, check_multiple_param_sets, fuzz


def test_exponential(plot_failures):
    """Test for exponential.Exponential"""
    x = np.linspace(0, 2, 100)
    params = {
        "x_dead": [0, 0.25],
        "y0": [-1, 0, 1],
        "y_inf": [-5, 5],
        "tau": [0.5, 1, 5],
    }
    model = Exponential()
    model.parameters["x_dead"].fixed_to = None
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )


def fuzz_exponential(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-2, 2, 100)
    fuzzed_params = {
        "x_dead": (0, 0.5),
        "y0": (-1, 1),
        "y_inf": (2, 5),
        "tau": (0.5, 5),
    }

    model = Exponential()
    model.parameters["x_dead"].fixed_to = None

    return fuzz(
        x=x,
        model=model,
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
