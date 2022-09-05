import logging
from matplotlib import pyplot as plt
import numpy as np
import pprint
import traceback
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import warnings

import fits


if TYPE_CHECKING:
    num_samples = float
    num_values = float


# TODO: how should we handle logging during tests?
logger = logging.getLogger(__name__)
warnings.filterwarnings("error")  # Promote divide by zero etc to hard errors

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-v", "--verbose", default=0, action="count", help="increase logging level"
# )
# parser.add_argument(
#     "-q", "--quiet", default=0, action="count", help="decrease logging level"
# )
# args, _ = parser.parse_known_args()

# logging.getLogger().setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING + args.quiet * 10 - args.verbose * 10)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)


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


# TODO: the design here doesn't feel quite right. Setting fixed_params in the constructor
# but the parameter values in the tests feels odd. Not sure what the right way of handling
# this is however...
class ModelTest:
    """Base class for tests for individual models."""

    def __init__(
        self,
        x: fits.utils.ArrayLike[("num_samples",), np.float64],
        fitter: fits.common.FitBase,
        *,
        param_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
        fixed_params: Optional[Dict[str, Optional[float]]] = None,
        initial_values: Optional[Dict[str, float]] = None,
        param_tol: Optional[float] = 1e-3,
        significance_tol: Optional[float] = None,
        residual_tol: Optional[float] = None,
        plot_failures: bool = True,
    ):
        """
        :param x: x-axis dataset
        :param fitter: the fitter to test with
        :param param_bounds: parameter bounds dictionary passed to the fitter
        :param fixed_params: fixed parameter dictionary passed to the fitter
        :param initial_values: parameter initial value dictionary passed to the fitter
        :param significance_tol: optional tolerance to check fitted parameters against
        :param p_tol: optional tolerance to check fit significance against
        :param residual_tol: optional tolerance to check fit residuals against
        :param plot_failures: if `True` we plot the dataset/fit results for failed tests
        """
        self._x = np.asarray(x)
        self._fitter = fitter

        self.param_tol = param_tol
        self.significance_tol = significance_tol
        self.residual_tol = residual_tol
        self.plot_failures = plot_failures

    def check_single_param_set(self, test_params: Dict[str, float]):
        """Validates the fit for a single set of parameter values.

        :param test_params: dictionary of parameter values to test
        """
        if set(test_params.keys()) != set(self._fitter._model.get_parameters().keys()):
            raise ValueError("Test parameter sets must match the model parameters")

        y, y_err = self.generate_data(test_params)

        logger.debug(
            f"Testing {self._fitter._model.__class__.__name__} with dataset:\n"
            f"x={pprint.pformat(self._x, indent=4)}\n"
            f"y={pprint.pformat(y, indent=4)}"
        )


        fitted_params, fitted_param_err = self._fitter.fit()

        params_str = pprint.pformat(test_params, indent=4)
        fitted_params_str = pprint.pformat(fitted_params, indent=4)

        if self.param_tol is not None and not params_close(
            test_params, fitted_params, self.param_tol
        ):
            self.plot_failure(y, y_err)
            raise ValueError(
                "Error in parameter values is too large:\n"
                f"test parameter set was: {params_str}\n"
                f"fitted parameters were: {fitted_params_str}"
            )

        if (
            self.significance_tol is not None
            and (p_fit := self._fitter.fit_significance()) < self.significance_tol
        ):
            self.plot_failure(y, y_err)
            raise ValueError(
                f"Fit significance too low: {p_fit:.2f} < {self.p_thresh:.2f}",
            )

        if self.residual_tol is not None and not is_close(
            y, self._fitter.evaluate(self._x)[1], self.residual_tol
        ):
            self.plot_failure()
            raise ValueError(
                "Fitted data not close to model:\n"
                f"actual parameter set was {params_str}\n"
                f"fitted parameters were: {fitted_params_str}"
            )

    def generate_data(
        self, params: Dict[str, float]
    ) -> Tuple[
        fits.utils.Array[("num_samples",), np.float64],
        Optional[fits.utils.ArrayLike[("num_samples",), np.float64]],
    ]:
        """Generates a synthetic dataset for a given set of parameter values.

        :param params: parameter dictionary
        :returns: tuple of y-axis values and their uncertainties
        """
        y = self._fitter._model.func(self._x, params)
        self._fitter.set_dataset(self._x, y)
        return y, None

    def plot_failure(self, y, y_err):
        if not self.plot_failures:
            return

        if y_err is None:
            plt.plot(self._x, y, "-o", label="data")
        else:
            plt.errorbar(self._x, y, "-o", yerr=y_err, label="data")

        plt.plot(self._x, self._fitter.evaluate(self._x)[1], "-.o", label="fit")
        plt.plot(
            self._x,
            self._fitter._model.func(self._x, self._fitter._estimated_values),
            "--o",
            label="heuristic",
        )
        plt.grid()
        plt.legend()
        plt.show()

    def check_multiple_params(
        self,
        test_params: Dict[str, fits.utils.ArrayLike[("num_values",), np.float64]],
    ):
        """Validates the fit for multiple sets of parameter values.

        :param test_params: dictionary of parameter values to test. Each entry may
            contain either a single value or an array of values to test.
        """

        model_params = set(self._fitter._model.get_parameters().keys())
        input_params = set(test_params.keys())
        fixed_params = set(self._fitter._fixed_params)

        assert not input_params.intersection(fixed_params), (
            f"Input parameters must not include fixed parameters")

        assert input_params.union(fixed_params) == model_params, (
            f"Input parameters '{input_params}' don't match model parameters "
            f"'{model_params}'"
        )



        test_params = dict(test_params)
        test_params.update(self._fitter._fixed_params)

        def walk_params(scanned_params, fixed_params):
            scanned_params = dict(scanned_params)
            fixed_params = dict(fixed_params)

            param, values = scanned_params.popitem()

            if np.isscalar(values):
                values = [values]

            for value in values:
                fixed_params[param] = value
                if scanned_params != {}:
                    walk_params(scanned_params, fixed_params)
                else:
                    self.check_single_param_set(fixed_params)

        walk_params(test_params, {})

    def fuzz(
        self,
        x: fits.utils.ArrayLike[("num_samples",), np.float64],
        static_params: Dict[str, float],
        fuzzed_params: Dict[str, Tuple[float, float]],
        fixed_params: Optional[Dict[str, float]] = None,
        num_trials: int = 100,
        param_tol: Optional[float] = 1e-3,
        p_thresh: Optional[float] = None,
        residual_tol: Optional[float] = None,
        plot_failures: bool = None,
        stop_at_failure: bool = True,
        **kwargs: Any,
    ):
        """Validates the fit for a single set of x-axis data and multiple
        randomly-generated sets of parameter values.

        :param x: x-axis points
        :param static_params: dictionary mapping names of model parameters which are
            evaluated with a single value to those values
        :param fuzzewd_params: dictionary mapping names of fuzzed model parameters
            to a tuple of `(lower, upper)` bounds to fuzz over. Parameter sets are
            randomly generated using a uniform distribution.
        :param fixed_params: dictionary mapping model parameter names to names that the
            fit is told to hold those values constant at. If `None` this is set to
            :param static_params:. *Fuzzed parameters are floated unless explicitly
            fixed in :param fixed_params:, even if the model fixes them by default.*
        :param param_tol: optional tolerance to check fitted parameters against.
        :param p_thresh: optional tolerance to check fit significance against.
        :param residual_tol: optional tolerance to check fit residuals against.
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails. If `None` we use the value set in :meth setUp`.
        :param stop_at_failure: if True we stop fuzzing the first time a test fails.
        :param kwargs: keyword arguments are passed into :meth test_single:
        """
        model_params = set(self.model.get_parameters().keys())
        fixed_params = dict(fixed_params if fixed_params is not None else static_params)

        fixed = set(fixed_params.keys())
        static = set(static_params.keys())
        fuzzed = set(fuzzed_params.keys())
        input_params = static.union(fuzzed)

        assert fuzzed.intersection(static) == set()
        assert input_params == model_params, (
            f"Input parameters '{input_params}' don't match model parameters "
            f"'{model_params}'"
        )
        assert set(fixed).intersection(fuzzed) == set(), (
            "Parameters cannot be both fixed and fuzzed: "
            f"{set(fixed).intersection(fuzzed)}"
        )
        fixed_params.update({param: fixed_params.get(param, None) for param in fuzzed})

        params = dict(fixed_params)
        failures = 0
        for trial in range(num_trials):
            for param, bounds in fuzzed_params.items():
                params[param] = np.random.uniform(*bounds)

            try:
                self.check_single(
                    x,
                    params,
                    fixed_params=fixed_params,
                    param_tol=param_tol,
                    p_thresh=p_thresh,
                    residual_tol=residual_tol,
                    plot_failures=plot_failures,
                    **kwargs,
                )

            except Exception:
                if stop_at_failure:
                    raise
                failures += 1
                logger.warning(f"failed...{traceback.format_exc()}")
        return failures


class NormalModelTest(ModelTest):
    """Test for fit model using `NormalFitter`."""

    def __init__(
        self,
        x: fits.utils.ArrayLike[("num_samples",), np.float64],
        model: fits.FitModel,
        *,
        param_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
        fixed_params: Optional[Dict[str, Optional[float]]] = None,
        initial_values: Optional[Dict[str, float]] = None,
        param_tol: Optional[float] = 1e-3,
        significance_tol: Optional[float] = None,
        residual_tol: Optional[float] = None,
        plot_failures: bool = True,
    ):
        """
        :param x: x-axis dataset
        :param model: the fit model to be tested
        :param param_bounds: parameter bounds dictionary passed to the fitter
        :param fixed_params: fixed parameter dictionary passed to the fitter
        :param initial_values: parameter initial value dictionary passed to the fitter
        :param significance_tol: optional tolerance to check fitted parameters against
        :param p_tol: optional tolerance to check fit significance against
        :param residual_tol: optional tolerance to check fit residuals against
        :param plot_failures: if `True` we plot the dataset/fit results for failed tests
        """
        fitter = fits.NormalFit(
            model=model,
            param_bounds=param_bounds,
            fixed_params=fixed_params,
            initial_values=initial_values,
        )

        super().__init__(
            x,
            fitter,
            param_bounds=param_bounds,
            fixed_params=fixed_params,
            initial_values=initial_values,
            param_tol=param_tol,
            significance_tol=significance_tol,
            residual_tol=residual_tol,
            plot_failures=plot_failures,
        )
