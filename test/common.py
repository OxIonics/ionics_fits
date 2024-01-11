import dataclasses
import logging
from matplotlib import pyplot as plt
import numpy as np
import pprint
import traceback
from typing import Callable, Dict, Optional, List, Tuple, Type, TYPE_CHECKING

import ionics_fits as fits


if TYPE_CHECKING:
    num_samples = float
    num_values = float


logger = logging.getLogger(__name__)


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
        :param_tol: tolerance to check fitted parameter values against
        :param residual_tol: tolerance to check fit residuals against
        :param heuristic_tol: tolerance to check heuristic values against
        :param plot_failures: if `True` we plot the dataset/fit results for failed tests
        :param plot_all: if `True` we plot every run, not just failures (used in
            debugging)
    """

    param_tol: Optional[float] = 1e-3
    residual_tol: Optional[float] = None
    heuristic_tol: Optional[float] = None
    plot_failures: bool = True
    plot_all: bool = False


def check_single_param_set(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    test_params: Dict[str, float],
    config: Optional[TestConfig] = None,
    fitter_cls: Optional[Type[fits.common.Fitter]] = fits.normal.NormalFitter,
    fitter_args: Optional[Dict] = None,
    user_estimates: Optional[List[str]] = None,
):
    """Validates the fit for a single set of parameter values.

    :param x: x-axis dataset
    :param model: the model to test.
    :param test_params: dictionary of parameter values to test
    :param test_config: test configuration
    :param fitter_cls: the fitter class to test with. Defaults to `NormalFitter`
    :param fitter_args: optional dictionary of kwargs to pass to the fitter class
    :param user_estimates: list of names of parameters to supply initial
        guesses for
    """
    fitter_cls = fitter_cls if fitter_cls is not None else fits.normal.NormalFitter

    x = np.asarray(x)
    config = config or TestConfig()

    if set(test_params.keys()) != set(model.parameters.keys()):
        raise ValueError(
            f"Test parameter set {set(test_params.keys())} must match the "
            f"model parameters {set(model.parameters.keys())}"
        )

    y = model.func(x, test_params)

    logger.debug(
        f"Testing {model.__class__.__name__} with dataset:\n"
        f"x={pprint.pformat(x, indent=4)}\n"
        f"y={pprint.pformat(y, indent=4)}"
    )

    if user_estimates is not None:
        for parameter in user_estimates:
            model.parameters[parameter].user_estimate = test_params[parameter]

    try:
        if fitter_args is not None:
            fit = fitter_cls(x=x, y=y, model=model, **fitter_args)
        else:
            fit = fitter_cls(x=x, y=y, model=model)

    except RuntimeError as ex:
        raise RuntimeError(
            f"{model.__class__.__name__} fit failed! Parameters were:\n"
            f"{pprint.pformat(test_params, indent=4)}\n"
        ) from ex

    fit.values = {param: fit.values[param] for param in fit.model.parameters.keys()}

    params_str = pprint.pformat(test_params, indent=4)
    fitted_params_str = pprint.pformat(fit.values, indent=4)
    initial_params_str = pprint.pformat(fit.initial_values, indent=4)

    if config.param_tol is not None and not params_close(
        test_params, fit.values, config.param_tol
    ):
        if config.plot_failures:
            _plot(
                fit,
                y,
            )

        raise ValueError(
            "Error in parameter values is too large:\n"
            f"test parameter set was: {params_str}\n"
            f"fitted parameters were: {fitted_params_str}\n"
            f"estimated parameters were: {initial_params_str}\n"
            f"free parameters were: {fit.free_parameters}"
        )

    if config.heuristic_tol is not None and not params_close(
        test_params, fit.initial_values, config.heuristic_tol
    ):
        if config.plot_failures:
            _plot(
                fit,
                y,
            )

        raise ValueError(
            "Error in heuristics is too large:\n"
            f"test parameter set was: {params_str}\n"
            f"estimated parameters were: {initial_params_str}\n"
            f"free parameters were: {fit.free_parameters}"
        )

    if config.residual_tol is not None and not is_close(
        y, fit.evaluate()[1], config.residual_tol
    ):
        if config.plot_failures:
            _plot(
                fit,
                y,
            )

        raise ValueError(
            "Error in residuals is too large:\n"
            f"test parameter set was: {params_str}\n"
            f"fitted parameters were: {fitted_params_str}\n"
            f"estimated parameters were: {initial_params_str}\n"
            f"free parameters were: {fit.free_parameters}"
        )

    if config.plot_all:
        _plot(fit, y)


def _plot(
    fit: fits.common.Fitter,
    y_model: fits.utils.Array[("num_samples",), np.float64],
):

    y_model = y_model
    y_fit = fit.model.func(fit.x, fit.values)
    y_heuristic = fit.model.func(fit.x, fit.initial_values)

    y_model = np.atleast_2d(y_model)
    y_heuristic = np.atleast_2d(y_heuristic)
    y_fit = np.atleast_2d(y_fit)

    _, ax = plt.subplots(fit.model.get_num_y_channels(), 2)
    for ch in range(fit.model.get_num_y_channels()):

        y_model_ch = y_model[ch, :]
        y_fit_ch = y_fit[ch, :]
        y_heuristic_ch = y_heuristic[ch, :]

        if fit.model.get_num_y_channels() == 1:
            ax = np.expand_dims(ax, axis=0)

        ax[ch, 0].set_title(fit.model.__class__.__name__)
        ax[ch, 0].plot(fit.x, y_model_ch, "-o", label="model")
        ax[ch, 0].plot(fit.x, y_fit_ch, "-.o", label="fit")
        ax[ch, 0].plot(
            fit.x,
            y_heuristic_ch,
            "--o",
            label="heuristic",
        )
        ax[ch, 0].set(
            xlabel="x", ylabel="y" if fit.model.get_num_y_channels() == 1 else f"y_{ch}"
        )
        ax[ch, 0].grid()
        ax[ch, 0].legend()

        freq_model, y_f_model = fits.models.heuristics.get_spectrum(fit.x, y_model_ch)
        freq_fit, y_f_fit = fits.models.heuristics.get_spectrum(fit.x, y_fit_ch)
        freq_heuristic, y_f_heuristic = fits.models.heuristics.get_spectrum(
            fit.x, y_heuristic_ch
        )

        ax[ch, 1].set_title(f"{fit.model.__class__.__name__} spectrum")
        ax[ch, 1].plot(freq_model, np.abs(y_f_model), "-o", label="model")
        ax[ch, 1].plot(freq_fit, np.abs(y_f_fit), "-.o", label="fit")
        ax[ch, 1].plot(
            freq_heuristic,
            np.abs(y_f_heuristic),
            "--o",
            label="heuristic",
        )
        ax[ch, 1].set(xlabel="frequency (linear units)", ylabel="Spectral density")
        ax[ch, 1].grid()
        ax[ch, 1].legend()

    plt.show()


def check_multiple_param_sets(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    test_params: Dict[str, List[float]],
    config: Optional[TestConfig] = None,
    fitter_cls: Type[fits.common.Fitter] = fits.normal.NormalFitter,
    fitter_args: Optional[Dict] = None,
    user_estimates: Optional[List[str]] = None,
):
    """Validates the fit for multiple sets of parameter values.

    :param x: x-axis dataset
    :param model: the model to test.
    :param test_params: dictionary of parameter values to test. Each entry may
        contain either a single value or an array of values to test.
    :param test_config: test configuration
    :param fitter_cls: the fitter class to test with
    :param fitter_args: optional dictionary of kwargs to pass to the fitter class
    :param user_estimates: optional list of names of parameters to supply
        initial guesses for
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

    def walk_params(remaining, scanned):
        remaining = dict(remaining)
        scanned = dict(scanned)

        param, values = remaining.popitem()

        if np.isscalar(values):
            values = [values]

        for value in values:
            scanned[param] = value
            if remaining != {}:
                walk_params(remaining, scanned)
            else:
                check_single_param_set(
                    x=x,
                    model=model,
                    test_params=scanned,
                    config=config,
                    fitter_cls=fitter_cls,
                    user_estimates=user_estimates,
                )

    walk_params(test_params, {})


def fuzz(
    x: fits.utils.ArrayLike[("num_samples",), np.float64],
    model: fits.common.Model,
    static_params: Dict[str, float],
    fuzzed_params: Dict[str, Tuple[float, float]],
    test_config: Optional[TestConfig] = None,
    fitter_cls: Optional[Type[fits.common.Fitter]] = fits.normal.NormalFitter,
    fitter_args: Optional[Dict] = None,
    num_trials: int = 100,
    stop_at_failure: bool = True,
    param_generator: Optional[
        Callable[[Dict[str, Tuple[float, float]]], Dict[str, float]]
    ] = None,
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
    :param fitter_args: optional dictionary of kwargs to pass to the fitter class
    :param num_trials: number of random parameter sets to test
    :param stop_at_failure: if True we stop fuzzing the first time a test fails.
    :param param_generator: Callable that takes a dictionary of fuzzed parameters and
        returns a dictionary mapping names of fuzzed parameters to randomly generated
        values. If `None` we use independent uniform distributions for all parameters.
    :returns: the number of failed runs
    """
    param_generator = param_generator or generate_param_set
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
        test_params.update(param_generator(fuzzed_params))

        try:
            check_single_param_set(
                x=x,
                model=model,
                test_params=test_params,
                config=test_config,
                fitter_cls=fitter_cls,
                fitter_args=fitter_args,
            )

        except Exception:
            if stop_at_failure:
                raise
            failures += 1
            logger.warning(f"failed...{traceback.format_exc()}")

    return failures


def generate_param_set(
    fuzzed_params: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    return {
        param: np.random.uniform(*bounds) for param, bounds in fuzzed_params.items()
    }
