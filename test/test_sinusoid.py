import numpy as np

from ionics_fits.models.sinusoid import Sine2, SineMinMax, Sinusoid

from .common import Config, check_multiple_param_sets, check_single_param_set, fuzz


def test_sinusoid(plot_failures: bool):
    """Test for sinusoid.Sinusoid"""
    x = np.linspace(-10, 10, 1000)
    params = {
        "a": 2,
        "omega": [3 / (2 * np.pi), 5 / (2 * np.pi), 10 / (2 * np.pi)],
        "phi": 0.5,
        "y0": 1,
        "x0": 0,
        "tau": np.inf,
    }
    model = Sinusoid()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.45),
    )


def test_sine_min_max(plot_failures: bool):
    """Test for sinusoid.SineMinMax"""
    x = np.linspace(-10, 10, 1000)
    params = {
        "min": -1,
        "max": 3,
        "omega": 10 / (2 * np.pi),
        "phi": 0.5,
        "x0": 0,
        "tau": np.inf,
    }
    model = SineMinMax()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.45),
    )


def test_sinusoid_heuristic(plot_failures: bool):
    """Check that the sinusoid heuristic gives an accurate estimate in an easy case"""
    x = np.linspace(-10, 10, 1000)
    params = {
        "a": 2,
        "omega": 10 / (2 * np.pi),
        "phi": 0.5,
        "y0": 1,
        "x0": 0,
        "tau": np.inf,
    }
    model = Sinusoid()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.02),
    )


def test_sine2_heuristic(plot_failures: bool):
    """Check that the Sine2 heuristic gives an accurate estimate in an easy case"""
    x = np.linspace(-10, 10, 1000)
    params = {
        "a": 2,
        "omega": 10 / (2 * np.pi),
        "phi": 0.75,
        "y0": 1,
        "x0": 0,
        "tau": np.inf,
    }

    model = Sine2()
    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0.075),
    )

    # Check the parameter estimator transfers user estimates across to its internal
    # sin-2x model
    model = Sine2()
    for param_name, param_value in params.items():
        if model.parameters[param_name].fixed_to is None:
            model.parameters[param_name].user_estimate = param_value

    check_single_param_set(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=0),
    )


def test_sinusoid_x0(plot_failures: bool):
    """Test for sinusoid.Sinusoid with `x0` floated instead of `phi`"""
    x = np.linspace(-5, 20, 1000)
    params = {
        "a": 2,
        "omega": 1 / (2 * np.pi),
        "phi": 0,
        "y0": 1,
        "x0": 1,
        "tau": np.inf,
    }
    model = Sinusoid()
    model.parameters["x0"].fixed_to = None
    model.parameters["phi"].fixed_to = 0
    check_single_param_set(x, model, params, Config(plot_failures=plot_failures))


def fuzz_sinusoid(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-2, 4, 1000)

    dx = np.ptp(x) / x.size
    w_nyquist = 0.5 * (1 / dx) * 2 * np.pi
    w_min = 1 / np.ptp(x) * 2 * np.pi

    static_params = {"tau": np.inf, "x0": 0}
    fuzzed_params = {
        "a": (0, 3),
        "omega": (2 * w_min, 0.5 * w_nyquist),
        "phi": (-np.pi, np.pi),
        "y0": (-3, 3),
    }
    # don't fail if a phase of ~ pi fits to ~ -pi
    test_config.param_tol = None
    test_config.residual_tol = 1e-3

    model = Sinusoid()
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
