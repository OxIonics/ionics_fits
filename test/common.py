import logging
from matplotlib import pyplot as plt
import numpy as np
import pprint
import traceback
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Type
import unittest
import warnings

import fits


if TYPE_CHECKING:
    num_samples = float
    num_values = float


# TODO: how should we handle logging during tests?
logger = logging.getLogger(__name__)

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


class TestBase(unittest.TestCase):
    def setUp(
        self,
        model: fits.FitModel,
        fit_class: Type[fits.FitBase] = fits.NormalFit,
        plot_failures: bool = False,
    ):
        """Call before running any tests.

        :param model_class: the model class to be tested
        :param fit_class: the fit class to be used during the test
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails
        """
        self.model = model
        self.fit_class = fit_class
        self.plot_failures = plot_failures

        warnings.filterwarnings("error")  # Promote divide by zero etc to hard errors

    def set_dataset(
        self,
        x: fits.utils.Array[("num_samples",), np.float64],
        params: Dict[str, float],
        fit: fits.FitBase,
    ) -> Tuple[
        fits.utils.Array[("num_samples",), np.float64],
        Optional[fits.utils.Array[("num_samples",), np.float64]],
    ]:
        """Generates a synthetic dataset at the given x-axis points for the given model
        parameters and passes it into the fit function.

        :param x: x-axis points
        :param params: dictionary mapping names of model parameters to their values
        :returns: dataset y-axis values and, optionally, their error bars
        """
        y = self.model.func(x, params)
        fit.set_dataset(x, y)
        return y, None

    def check_single(
        self,
        x: fits.utils.ArrayLike[("num_samples",), np.float64],
        params: Dict[str, float],
        param_tol: Optional[float] = 1e-3,
        p_thresh: Optional[float] = None,
        residual_tol: Optional[float] = None,
        plot_failures: bool = None,
        **kwargs: Any,
    ):
        """Validates the fit for a single set of x-axis data and parameter values.

        :param x: x-axis points
        :param params: dictionary mapping names of model parameters to their values
        :param param_tol: optional tolerance to check fitted parameters against.
        :param p_thresh: optional tolerance to check fit significance against.
        :param residual_tol: optional tolerance to check fit residuals against.
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails. If `None` we use the value set in :meth setUp`.
        :param kwargs: keyword arguments are passed directly into the fit class
            constructor.
        """
        if set(params.keys()) != set(self.model.get_parameters().keys()):
            raise ValueError("Test parameter sets must match the model parameters")

        fit = self.fit_class(self.model, **kwargs)
        y, y_err = self.set_dataset(x, params, fit)
        fitted_params, fitted_param_err = fit.fit()

        logger.debug(
            "Testing with dataset:\n"
            f"x={pprint.pformat(x, indent=4)}\n"
            f"y={pprint.pformat(y, indent=4)}"
        )

        def plot():
            do_plot = plot_failures if plot_failures is not None else self.plot_failures

            if not do_plot:
                return

            plt.title(self.__class__.__name__)
            if y_err is None:
                plt.plot(x, y, "-o", label="data")
            else:
                plt.errorbar(x, y, "-o", yerr=y_err, label="data")

            plt.plot(x, fit.evaluate(x)[1], "-.o", label="fit")
            plt.plot(
                x, fit._model.func(x, fit._estimated_values), "--o", label="heuristic"
            )
            plt.grid()
            plt.legend()
            plt.show()

        params_str = pprint.pformat(params, indent=4)
        fitted_params_str = pprint.pformat(fitted_params, indent=4)

        if param_tol is not None:
            if not self.params_close(params, fitted_params, param_tol):
                plot()
                raise ValueError(
                    "Error in parameter values is too large:\n"
                    f"actual parameter set was {params_str}\n"
                    f"fitted parameters were: {fitted_params_str}"
                )

        if p_thresh is not None:
            p_fit = fit.fit_significance()

            if p_fit < self.p_thresh:
                plot()

                raise ValueError(
                    f"Fit significance too low: {p_fit:.2f} < {self.p_thresh:.2f}",
                )

        if residual_tol is not None:
            if not self.is_close(y, fit.evaluate(x)[1], residual_tol):
                plot()
                raise ValueError(
                    "Fitted data not close to model:\n"
                    f"actual parameter set was {params_str}\n"
                    f"fitted parameters were: {fitted_params_str}"
                )

    @staticmethod
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

    @staticmethod
    def params_close(
        nominal_params: Dict[str, float], fitted_params: Dict[str, float], tol: float
    ):
        """Returns true if the values of all fitted parameters are close to the nominal
        values.
        """
        return all(
            [
                TestBase.is_close(nominal_params[param], fitted_params[param], tol)
                for param in nominal_params.keys()
            ]
        )

    def check_multiple(
        self,
        x: fits.utils.ArrayLike[("num_samples",), np.float64],
        static_params: Dict[str, float],
        scanned_params: Dict[str, fits.utils.ArrayLike[("num_values",), np.float64]],
        fixed_params: Optional[Dict[str, float]] = None,
        param_tol: Optional[float] = 1e-3,
        p_thresh: Optional[float] = None,
        residual_tol: Optional[float] = None,
        plot_failures: bool = None,
        **kwargs: Any,
    ):
        """Validates the fit for a single set of x-axis data and multiple sets of
        parameter values.

        :param x: x-axis points
        :param static_params: dictionary mapping names of model parameters which are
            evaluated with a single value to those values
        :param scanned_params: dictionary mapping names of model parameters which are
            evaluated at multiple values to arrays of those values.
        :param fixed_params: dictionary mapping model parameter names to names that the
            fit is told to hold those values constant at. If `None` this is set to
            :param static_params:.  *Scanned parameters are floated unless explicitly
            fixed in :param fixed_params:, even if the model fixes them by default.*
        :param param_tol: optional tolerance to check fitted parameters against.
        :param p_thresh: optional tolerance to check fit significance against.
        :param residual_tol: optional tolerance to check fit residuals against.
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails. If `None` we use the value set in :meth setUp`.
        :param kwargs: keyword arguments are passed into :meth test_single:
        """

        model_params = set(self.model.get_parameters().keys())
        fixed_params = dict(fixed_params if fixed_params is not None else static_params)

        static = set(static_params.keys())
        scanned = set(scanned_params.keys())
        fixed = set(fixed_params.keys())
        input_params = static.union(scanned)

        assert scanned.intersection(static) == set()
        assert input_params == model_params, (
            f"Input parameters '{input_params}' don't match model parameters "
            f"'{model_params}'"
        )
        assert set(fixed).intersection(scanned) == set(), (
            "Parameters cannot be both fixed and scanned: "
            f"{set(fixed).intersection(scanned)}"
        )
        fixed_params.update({param: fixed_params.get(param, None) for param in scanned})

        params = dict(static_params)

        def walk_params(params, static, scanned):
            scanned = dict(scanned)
            param, values = scanned.popitem()
            for value in values:
                params[param] = value
                if scanned:
                    walk_params(params, static, scanned)
                else:
                    self.check_single(
                        x,
                        params,
                        param_tol=param_tol,
                        p_thresh=p_thresh,
                        residual_tol=residual_tol,
                        plot_failures=plot_failures,
                        fixed_params=fixed_params,
                        **kwargs,
                    )

        walk_params(params, fixed_params, scanned_params)

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


class TestPoisson(TestBase):
    """Test using Possonian statistics to generate the synthetic datasets.

    NB we do not currently have a Possonian MLE fitter so we approximate with a
    least-squares fit. Tests based on fit significance are prone to failure
    because the Poisson distribution has a longer tail than a normal distribution.
    """

    def setUp(
        self,
        model: fits.FitModel,
        plot_failures: bool = False,
    ):
        """Call before running any tests.

        :param model: the model class to be tested
        :param fit_class: the fit class to be used during the test
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails
        """
        super().setUp(
            model=model,
            fit_class=fits.NormalFit,
            plot_failures=plot_failures,
        )

    def set_dataset(
        self,
        x: fits.utils.Array[("num_samples",), np.float64],
        params: Dict[str, float],
        fit: fits.FitBase,
    ) -> Tuple[
        fits.utils.Array[("num_samples",), np.float64],
        Optional[fits.utils.Array[("num_samples",), np.float64]],
    ]:
        """Generates a synthetic dataset at the given x-axis points for the given model
        parameters and passes it into the fit function.

        :param x: x-axis points
        :param params: dictionary mapping names of model parameters to their values
        :returns: dataset y-axis values and their error bars
        """
        y_model = super().set_dataset(x, params)[0]

        if any(y_model < 0):
            raise ValueError("The mean of a Possonian variable must be >0")

        y = np.random.poisson(y_model)
        y_err = np.sqrt(y_model)  # std = sqrt(mean) for Poisson

        return y, y_err


class TestBinomial(TestBase):
    """Test using Binomial statistics to generate the synthetic datasets.

    NB we do not currently have a Binomial MLE fitter so we approximate with a
    least-squares fit. This is accurate in the limit of large number of shots.
    """

    def setUp(
        self,
        model: fits.FitModel,
        plot_failures: bool = False,
        num_shots: int = 100,
    ):
        """Call before running any tests.

        :param model: the model class to be tested
        :param plot_failures: if `True` we plot the dataset/fit results each time the
            test fails
        :param num_shots: number of shots used at each data point. May be changed later
            on using the `num_shots` attribute.
        """
        super().setUp(
            model=model,
            fit_class=fits.NormalFit,
            plot_failures=plot_failures,
        )
        self.num_shots = num_shots

    def set_dataset(
        self,
        x: fits.utils.Array[("num_samples",), np.float64],
        params: Dict[str, float],
        fit: fits.FitBase,
    ) -> Tuple[
        fits.utils.Array[("num_samples",), np.float64],
        Optional[fits.utils.Array[("num_samples",), np.float64]],
    ]:
        """Generates a synthetic dataset at the given x-axis points for the given model
        parameters and passes it into the fit function.

        :param x: x-axis points
        :param params: dictionary mapping names of model parameters to their values
        :returns: dataset y-axis values and their error bars
        """
        y_model = super().set_dataset(x, params)[0]

        if any(0 > y_model > 1):
            raise ValueError("The mean of a Binomial variable must lie between 0 and 1")

        y = np.random.binomial(self.num_shots, y_model)

        # TODO: this is an approximation for when we're using NormalFits
        # q = 1 - y_model  # For BinomialFit
        # y_err = np.sqrt(self.num_shots * y_model * q)  # For BinomialFit
        y_err = fits.utils.binom_onesided(
            np.array(y_model * self.num_shots, dtype=int), self.num_shots
        )

        return y, y_err
