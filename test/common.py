import dataclasses
import logging
from matplotlib import pyplot as plt
import numpy as np
import pprint
import traceback
from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING
import warnings

import ionics_fits as fits


if TYPE_CHECKING:
    num_samples = float
    num_values = float


logger = logging.getLogger(__name__)
warnings.filterwarnings("error")  # Promote divide by zero etc to hard errors


def is_close(
    a: fits.utils.ArrayLike[("num_samples",), np.float64],
    b: fits.utils.ArrayLike[("num_samples",), np.float64],
    tol: float,
):
    """Returns True if `a` and `b` are approximately equal.

    `np.isclose` computes `(|a-b| <= atol + rtol * |b|)`, but what we really want is
    `|a-b| <= atol` OR `|a-b| <= rtol * |b|`.
    """
    return np.all(
        np.logical_or(
            np.isclose(a, b, rtol=tol, atol=0), np.isclose(a, b, rtol=0, atol=tol)
        )
    )


def params_close(
    nominal_params: Dict[str, float], fitted_params: Dict[str, float], tol: float
):
    """Returns true if the values of all fitted parameters are close to the nominal
    values.
    """
    return all(
        [
            is_close(nominal_params[param], fitted_params[param], tol)
            for param in nominal_params.keys()
        ]
    )


@dataclasses.dataclass
class TestConfig:
    """Configuration settings for a model test.

    Attributes (test are only performed if values are not `None`):
        :param significance_tol: tolerance to check fitted parameters against
        :param p_tol: tolerance to check fit significance against
        :param residual_tol: tolerance to check fit residuals against
        :param plot_failures: if `True` we plot the dataset/fit results for failed tests
    """

    param_tol: Optional[float] = 1e-3
    significance_tol: Optional[float] = None
    residual_tol: Optional[float] = None
    plot_failures: bool = True


def check_single_param_set(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    test_params: Dict[str, float],
    config: Optional[TestConfig] = None,
    fitter_cls: Optional[Type[fits.common.Fitter]] = fits.normal.NormalFitter,
):
    """Validates the fit for a single set of parameter values.

    :param x: x-axis dataset
    :param model: the model to test.
    :param test_params: dictionary of parameter values to test
    :param test_config: test configuration
    :param fitter_cls: the fitter class to test with. Defaults to `NormalFitter`.
    """
    fitter_cls = fitter_cls if fitter_cls is not None else fits.normal.NormalFitter

    x = np.asarray(x)
    config = config or TestConfig()

    if set(test_params.keys()) != set(model.parameters.keys()):
        raise ValueError("Test parameter sets must match the model parameters")

    y = model.func(x, test_params)

    logger.debug(
        f"Testing {model.__class__.__name__} with dataset:\n"
        f"x={pprint.pformat(x, indent=4)}\n"
        f"y={pprint.pformat(y, indent=4)}"
    )

    fit = fitter_cls(x=x, y=y, sigma=None, model=model)

    params_str = pprint.pformat(test_params, indent=4)
    fitted_params_str = pprint.pformat(fit.values, indent=4)

    if config.param_tol is not None and not params_close(
        test_params, fit.values, config.param_tol
    ):
        _plot_failure(fit, y, config)
        raise ValueError(
            "Error in parameter values is too large:\n"
            f"test parameter set was: {params_str}\n"
            f"fitted parameters were: {fitted_params_str}"
        )

    if (
        config.significance_tol is not None
        and fit.fit_significance is not None
        and fit.fit_significance < config.significance_tol
    ):
        _plot_failure(fit, y, config)
        raise ValueError(
            f"Fit significance too low: {fit.fit_significance:.2f} < "
            f"{config.p_thresh:.2f}",
        )

    if config.residual_tol is not None and not is_close(
        y, fit.evaluate()[1], config.residual_tol
    ):
        _plot_failure(fit, y, config)
        raise ValueError(
            "Fitted data not close to model:\n"
            f"actual parameter set was {params_str}\n"
            f"fitted parameters were: {fitted_params_str}"
        )


def _plot_failure(
    fit: fits.common.Fitter,
    y_model: fits.utils.Array[("num_samples",), np.float64],
    config: TestConfig,
):
    if not config.plot_failures:
        return

    plt.plot(fit.x, y_model, "-o", label="model")
    plt.plot(*fit.evaluate(), "-.o", label="fit")
    plt.plot(
        fit.x,
        fit.model.func(fit.x, fit.initial_values),
        "--o",
        label="heuristic",
    )
    plt.grid()
    plt.legend()
    plt.show()


def check_multiple_param_sets(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    test_params: Dict[str, float],
    config: Optional[TestConfig] = None,
    fitter_cls: Type[fits.common.Fitter] = fits.normal.NormalFitter,
):
    """Validates the fit for multiple sets of parameter values.

    :param x: x-axis dataset
    :param model: the model to test.
    :param test_params: dictionary of parameter values to test. Each entry may
        contain either a single value or an array of values to test.
    :param test_config: test configuration
    :param fitter_cls: the fitter class to test with
    """
    model_params = set(model.parameters.keys())
    input_params = set(test_params.keys())
    fixed_params = set(
        [
            param
            for param, param_data in model.parameters.items()
            if param_data.fixed_to is None
        ]
    )

    assert input_params.union(fixed_params) == model_params, (
        f"Input parameters ({input_params}) + fixed parameters ({fixed_params}) match "
        f"the set of model parameters ({model_params})"
    )

    test_params = dict(test_params)

    def walk_params(remaining, scaned):
        remaining = dict(remaining)
        scaned = dict(scaned)

        param, values = remaining.popitem()

        if np.isscalar(values):
            values = [values]

        for value in values:
            scaned[param] = value
            if remaining != {}:
                walk_params(remaining, scaned)
            else:
                check_single_param_set(
                    x=x,
                    model=model,
                    test_params=scaned,
                    config=config,
                    fitter_cls=fitter_cls,
                )

    walk_params(test_params, {})


def fuzz(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    static_params: Dict[str, float],
    fuzzed_params: Dict[str, Tuple[float, float]],
    test_config: Optional[TestConfig] = None,
    fitter_cls: Type[fits.common.Fitter] = fits.normal.NormalFitter,
    num_trials: int = 100,
    stop_at_failure: bool = True,
) -> float:
    """Validates the fit for a single set of x-axis data and multiple
    randomly-generated sets of parameter values.

    :param x: x-axis dataset
    :param model: the model to test.
    :param static_params: dictionary mapping names of model parameters which are
        evaluated with a single value to those values
    :param fuzzed_params: dictionary mapping names of fuzzed model parameters
        to a tuple of `(lower, upper)` bounds to fuzz over. Parameter sets are
        randomly generated between `lower` and `upper` according to a uniform
        distribution.
    :param test_config: test configuration
    :param fitter_cls: the fitter class to test with
    :param num_trials: number of random parameter sets to test
    :param stop_at_failure: if True we stop fuzzing the first time a test fails.
    :returns: the number of failed runs
    """
    fixed = set(
        [
            param
            for param, param_data in model.parameters.items()
            if param_data.fixed_to is not None
        ]
    )

    model_params = set(model.parameters.keys())
    static = set(static_params.keys())
    fuzzed = set(fuzzed_params.keys())
    input_params = static.union(fuzzed)

    assert (
        fuzzed.intersection(static) == set()
    ), f"Parameters must not be fuzzed and static: {fuzzed.intersection(static)}"

    assert input_params == model_params, (
        f"Input parameters '{input_params}' don't match model parameters "
        f"'{model_params}'"
    )

    assert set(fixed).intersection(fuzzed) == set(), (
        "Parameters cannot be both fixed and fuzzed: "
        f"{set(fixed).intersection(fuzzed)}"
    )

    failures = 0
    for trial in range(num_trials):
        test_params = dict(static_params)
        test_params.update(
            {
                param: np.random.uniform(*bounds)
                for param, bounds in fuzzed_params.items()
            }
        )

        try:
            check_single_param_set(
                x=x,
                model=model,
                test_params=test_params,
                config=test_config,
                fitter_cls=fitter_cls,
            )

        except Exception:
            if stop_at_failure:
                raise
            failures += 1
            logger.warning(f"failed...{traceback.format_exc()}")

    return failures
