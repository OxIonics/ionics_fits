import numpy as np

from ionics_fits.models.polynomial import Line, Parabola, Polynomial, Power
from .common import check_multiple_param_sets, check_single_param_set, fuzz, Config


def test_line(plot_failures: bool):
    """Simple test for polynomials.Line

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """
    x = np.linspace(-10, 10)
    params = {"a": 3.2, "y0": -9}
    model = Line()
    check_single_param_set(x, model, params, Config(plot_failures=plot_failures))


def test_parabola(plot_failures: bool):
    """Simple test for polynomials.Parabola

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """
    x = np.linspace(-10, 10)
    params = {"k": -9, "y0": +4, "x0": -3}
    model = Parabola()
    check_single_param_set(x, model, params, Config(plot_failures=plot_failures))


def test_power_n(plot_failures: bool):
    """Test polynomials.Power with `n` and `y0` floated.

    We use using parameter sets with various values of `n` and `y0`, holding `x0` and
    `a` are held at fixed values.
    """
    x = np.linspace(0.1, 1)
    params = {"x0": 0, "a": 1, "n": [-5.51, -1, 1, 0, 3], "y0": [0, 1e-5, -1e-3]}
    model = Power()
    model.parameters["x0"].fixed_to = params["x0"]
    model.parameters["a"].fixed_to = params["a"]
    check_multiple_param_sets(x, model, params, Config(plot_failures=plot_failures))


def test_power_a(plot_failures: bool):
    """Test polynomials.Power with `a` and `y0` floated.

    We use using parameter sets with various values of `a` and `y0`, holding `x0` and
    `n` are held at fixed values.
    """
    x = np.linspace(0.1, 15)
    params = {"x0": 0, "n": 5, "a": [0.5, 1, 3, 5.5], "y0": [-2.4, 0, 1, 100]}
    model = Power()
    model.parameters["a"].fixed_to = None
    model.parameters["x0"].fixed_to = params["x0"]
    model.parameters["n"].fixed_to = params["n"]
    check_multiple_param_sets(x, model, params, Config(plot_failures=plot_failures))


def fuzz_power(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(0.1, 15)

    static_params = {"x0": 0, "a": 1}
    fuzzed_params = {"n": (-3, 3), "y0": (-10, 10)}

    return fuzz(
        x=x,
        model=Power(),
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )


def test_polynomial(plot_failures: bool):
    """Test for polynomials.Polynomial."""
    x = np.linspace(-5, 5)
    params = {"x0": 0, "a_0": 1, "a_1": 50, "a_2": 10, "a_3": 2}
    model = Polynomial(poly_degree=3)
    check_single_param_set(x, model, params, Config(plot_failures=plot_failures))


def test_x0(plot_failures: bool):
    """Test for polynomials.Polynomial floating x0."""

    # Floating x0 leads to an under-defined problem so we check the residuals rather
    # than the parameter values.
    x = np.linspace(-5, 50) * 1e-3
    params = {"x0": -10, "a_0": 1, "a_1": 50, "a_2": 10}
    model = Polynomial(poly_degree=2)
    model.parameters["x0"].fixed_to = None
    config = Config(residual_tol=1e-6, param_tol=None, plot_failures=plot_failures)
    check_single_param_set(x, model, params, config=config)


def fuzz_polynomial(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-5, 50) * 1e-3

    static_params = {"x0": 0}
    fuzzed_params = {f"a_{n}": (-10, 10) for n in range(4)}

    return fuzz(
        x=x,
        model=Polynomial(poly_degree=3),
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
