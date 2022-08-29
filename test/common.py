import numpy as np
import logging
from matplotlib import pyplot as plt
import traceback
from typing import Any, Dict, Optional, Tuple, Union
import unittest
import warnings
import fits


# TODO: fixup logging during tests
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
        model_class: fits.FitModel,
        fit_class: fits.FitBase = fits.NormalFit,
        plot_failures=False,
    ):
        self.model_class = model_class
        self.fit_class = fit_class
        self.plot_failures = plot_failures

        warnings.filterwarnings("error")  # Promote divide by zero etc to hard errors

    def get_data(self, x, params):
        return self.model_class.func(x, params), None

    def _test_single(
        self,
        x: np.array,
        params: Dict[str, float],
        param_tol=1e-3,
        p_thresh=0.9,
        residual_tol=None,
        plot_failures=None,
        **kwargs: Any,
    ):
        """Validates the fit for a single set of x-axis data and parameter values.

        keyword arguments are passed directly into the fit class constructor.
        """
        if set(params.keys()) != set(self.model_class.get_parameters().keys()):
            raise ValueError("Test parameter sets must match the model parameters")

        y, y_err = self.get_data(x, params)

        fit = self.fit_class(self.model_class, **kwargs)
        fit.set_dataset(x, y, y_err)
        fitted_params, fitted_param_err = fit.fit()

        def plot():
            do_plot = plot_failures if plot_failures is not None else self.plot_failures

            if not do_plot:
                return

            plt.title(self.__class__.__name__)
            if y_err is None:
                plt.plot(x, y, "-o", label="data")
            else:
                plt.errorbar(x, y, "-o", yerr=y_err, label="data")

            plt.plot(x, fit.evaluate(x)[1], "--o", label="fit")
            plt.plot(
                x, fit._model.func(x, fit._estimated_values), "-o", label="heuristic"
            )
            plt.grid()
            plt.legend()
            plt.show()

        if param_tol is not None:
            for param in fit._free_params:
                if not self.params_close(param, params, fitted_params, param_tol):
                    plot()
                    # TODO: tidy up error message
                    raise ValueError(
                        f"Error in {param} is too large: actual value is "
                        f"{params[param]:.3e}"
                        f" fitted value is {fitted_params[param]:.3e}"
                        f" actual parameter set: {params}"
                        f" fitted parameter set: {fitted_params}"
                    )

        if p_thresh is not None and y_err is not None:
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
                    f"Fitted data not close to model: {y}, {fit.evaluate(x)[1]}"
                )

    @staticmethod
    def is_close(a, b, tol):
        # np.isclose computes (|a-b| <= atol + rtol * |b|), but what we really want is
        # |a-b| <= atol OR |a-b| <= rtol * |b|.
        return np.all(
            np.logical_or(
                np.isclose(a, b, rtol=tol, atol=0), np.isclose(a, b, rtol=0, atol=tol)
            )
        )

    @staticmethod
    def params_close(param, nominal_params, fitted_params, tol):
        return TestBase.is_close(nominal_params[param], fitted_params[param], tol)

    def _test_multiple(
        self,
        x,
        static_params: Dict[str, float],
        scanned_params: Dict[str, Union[float, np.array]],
        fixed_params: Optional[Dict[str, float]] = None,
        param_tol=1e-3,
        p_thresh=0.9,
        residual_tol=None,
        **kwargs,
    ):
        model_params = set(self.model_class.get_parameters().keys())
        static = set(static_params.keys())
        scanned = set(scanned_params.keys())
        input_params = static.union(scanned)

        assert scanned.intersection(static) == set()
        assert input_params == model_params, (
            f"Input parameters '{input_params}' don't match model parameters "
            f"'{model_params}'"
        )

        if fixed_params is None:
            fixed_params = {param: static_params.get(param) for param in model_params}

        params = dict(static_params)

        def walk_params(params, static, scanned):
            scanned = dict(scanned)
            param, values = scanned.popitem()
            for value in values:
                params[param] = value
                if scanned:
                    walk_params(params, static, scanned)
                else:
                    self._test_single(
                        x,
                        params,
                        fixed_params=fixed_params,
                        param_tol=param_tol,
                        p_thresh=p_thresh,
                        residual_tol=residual_tol,
                        **kwargs,
                    )

        walk_params(params, fixed_params, scanned_params)

    def fuzz(
        self,
        x,
        static_params: Dict[str, float],
        fuzzed_params: Dict[str, Tuple[float, float]],
        fixed_params: Optional[Dict[str, float]] = None,
        num_trials=10,
        param_tol=1e-3,
        p_thresh=0.9,
        residual_tol=None,
        stop_at_failure=True,
        plot_failures=False,
    ):
        model_params = set(self.model_class.get_parameters().keys())
        static = set(static_params.keys())
        fuzzed = set(fuzzed_params.keys())
        input_params = static.union(fuzzed)

        assert fuzzed.intersection(static) == set()
        assert input_params == model_params, (
            f"Input parameters '{input_params}' don't match model parameters "
            f"'{model_params}'"
        )

        if fixed_params is None:
            fixed_params = {param: static_params.get(param) for param in model_params}

        params = dict(fixed_params)
        failures = 0
        for trial in range(num_trials):
            for param, bounds in fuzzed_params.items():
                params[param] = np.random.uniform(*bounds)

            try:
                self._test_single(
                    x,
                    params,
                    fixed_params=fixed_params,
                    param_tol=param_tol,
                    p_thresh=p_thresh,
                    residual_tol=residual_tol,
                    plot_failures=plot_failures,
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
    least-squares fit. Tests based on fit significance are prone to failure for large
    y-values because the Poisson distribution has a longer tail than a normal
    distribution.
    """

    def setUp(
        self,
        model_class: fits.FitModel,
        plot_failures=False,
    ):
        # TODO: move over to PoissonFit
        super().setUp(
            model_class=model_class,
            fit_class=fits.NormalFit,
            plot_failures=plot_failures,
        )

    def get_data(self, x, params):
        y_model = super().get_data(x, params)[0]
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
        model_class: fits.FitModel,
        plot_failures=False,
        num_shots=100,
    ):
        # TODO: move over to BinomialFit
        super().setUp(
            model_class=model_class,
            fit_class=fits.NormalFit,
            plot_failures=plot_failures,
        )
        self.num_shots = num_shots

    def get_data(self, x, params):
        y_model = super().get_data(x, params)[0]

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
