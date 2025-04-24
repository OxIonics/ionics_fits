import numpy as np

from ionics_fits.models.rectangle import Rectangle

from .common import Config, check_multiple_param_sets, fuzz


def test_rectangle(plot_failures: bool):
    """Test for rectangle.Rectangle

    NB the residuals can be bad here as the optimizer sometimes struggles to find the
    correct value for `x_l` and `x_r` because the function does not change for steps
    of less than the x spacing.
    """
    x = np.linspace(-2, 2, 100)
    params = {
        "a": [-1, +1],
        "y0": [-1, +1],
        "x_l": -1,
        "x_r": [1, 2.01],
    }
    model = Rectangle()

    model.parameters["x_l"].fixed_to = params["x_l"]
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=5e-2),
    )

    params["x_l"] = -10
    params["x_r"] = 1
    model.parameters["x_l"].fixed_to = params["x_l"]
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=5e-2),
    )

    params["x_l"] = [-2.01, -1]
    params["x_r"] = 1
    model.parameters["x_l"].fixed_to = None
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=5e-2),
    )


def fuzz_rectangle(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-2, 2, 100)
    fuzzed_params = {
        "a": (-2, 2),
        "y0": (-1, +1),
        "x_r": (1, 2.01),
    }
    static_params = {"x_l": -1}

    model = Rectangle()
    model.parameters["x_l"].fixed_to = static_params["x_l"]

    test_config.param_tol = 5e-2

    return fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
