import numpy as np
import test
from typing import Optional
import fits


def test_line():
    """Simple test for polynomials.Line

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """
    x = np.linspace(-10, 10)
    params = {"a": 3.2, "y0": -9}
    model = fits.models.Line()
    test.common.check_single_param_set(x, model, params)


def test_parabola():
    """Simple test for polynomials.Parabola

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """
    x = np.linspace(-10, 10)
    params = {"k": -9, "y0": +4, "x0": -3}
    model = fits.models.Parabola()
    test.common.check_single_param_set(x, model, params)


def test_power_n():
    """Test polynomials.Power with `n` and `y0` floated.

    We use using parameter sets with various values of `n` and `y0`, holding `x0` and
    `a` are held at fixed values.
    """
    x = np.linspace(0.1, 1)
    params = {"x0": 0, "a": 1, "n": [-5.51, -1, 1, 0, 3], "y0": [0, 1e-5, -1e-3]}
    model = fits.models.Power()
    model.parameters["x0"].fixed_to = params["x0"]
    model.parameters["a"].fixed_to = params["a"]
    test.common.check_multiple_param_sets(x, model, params)


def test_power_a():
    """Test polynomials.Power with `a` and `y0` floated.

    We use using parameter sets with various values of `a` and `y0`, holding `x0` and
    `n` are held at fixed values.
    """
    x = np.linspace(0.1, 15)
    params = {"x0": 0, "n": 5, "a": [0.5, 1, 3, 5.5], "y0": [-2.4, 0, 1, 100]}
    model = fits.models.Power()
    model.parameters["a"].fixed_to = None
    model.parameters["x0"].fixed_to = params["x0"]
    model.parameters["n"].fixed_to = params["n"]
    test.common.check_multiple_param_sets(x, model, params)


def fuzz_power(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(0.1, 15)

    static_params = {"x0": 0, "a": 1}
    fuzzed_params = {"n": (-3, 3), "y0": (-10, 10)}
    model = fits.models.Power()
    return test.common.fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )


def test_polynomial():
    """Test for polynomials.Polynomial."""
    x = np.linspace(-5, 5)
    params = {"x0": 0, "a_0": 1, "a_1": 50, "a_2": 10, "a_3": 2}
    params.update({f"a_{n}": 0 for n in range(4, 11)})
    model = fits.models.Polynomial(poly_degree=10)
    model.parameters["a_2"].fixed_to = None
    model.parameters["a_3"].fixed_to = None
    test.common.check_single_param_set(x, model, params)


def test_x0():
    """Test for polynomials.Polynomial floating x0."""

    # Floating x0 leads to an under-defined problem so we check the residuals rather
    # than the parameter values.
    x = np.linspace(-5, 50) * 1e-3
    params = {"x0": -10, "a_0": 1, "a_1": 50, "a_2": 10}
    model = fits.models.Polynomial(poly_degree=2)
    model.parameters["x0"].fixed_to = None
    model.parameters["a_2"].fixed_to = None
    config = test.common.TestConfig(residual_tol=1e-6, param_tol=None)
    test.common.check_single_param_set(x, model, params, config=config)


def fuzz_polynomial(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[test.common.TestConfig] = None,
) -> float:
    x = np.linspace(-5, 50) * 1e-3

    static_params = {f"a_{n}": 0 for n in range(4, 11)}
    static_params.update({"x0": 0})

    fuzzed_params = {f"a_{n}": (-10, 10) for n in range(4)}

    model = fits.models.Polynomial(poly_degree=10)
    model.parameters["a_2"].fixed_to = None
    model.parameters["a_3"].fixed_to = None

    return test.common.fuzz(
        x=x,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )
