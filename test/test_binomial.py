import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from ionics_fits.binomial import BinomialFitter
from ionics_fits.models.gaussian import Gaussian
from ionics_fits.models.sinusoid import Sinusoid, SineMinMax
from ionics_fits.normal import NormalFitter
from .common import check_single_param_set, Config


def test_binomial(plot_failures):
    """Basic test of binomial fitting"""
    num_trials = 1000
    x = np.linspace(-3, 3, 200) * 2 * np.pi
    model = Sinusoid()
    params = {
        "a": 0.5,
        "omega": 1,
        "phi": 1,
        "y0": 0.5,
        "x0": 0,
        "tau": np.inf,
    }
    model.parameters["a"].fixed_to = params["a"]
    model.parameters["y0"].fixed_to = params["y0"]

    check_single_param_set(
        x=x,
        model=model,
        test_params=params,
        config=Config(plot_failures=plot_failures),
        fitter_cls=BinomialFitter,
        fitter_args={"num_trials": num_trials},
    )


def test_binomial_synthetic(plot_failures):
    """
    Check that the BinomialFitter gives an unbiased parameter estimate with correct
    parameter standard errors.
    """
    num_trials = 200
    num_datasets = 1000

    x = np.linspace(-1, 1, 200) * 2 * np.pi
    model = SineMinMax()
    params = {
        "min": 0.005,
        "max": 0.995,
        "omega": 1,
        "phi": 1,
        "x0": 0,
        "tau": np.inf,
    }

    model.parameters["omega"].fixed_to = params["omega"]
    model.parameters["x0"].fixed_to = params["x0"]

    model.parameters["min"].lower_bound = 0
    model.parameters["min"].upper_bound = 0.5
    model.parameters["max"].lower_bound = 0.5
    model.parameters["max"].upper_bound = 1

    model.parameters["omega"].lower_bound = 0
    model.parameters["omega"].upper_bound = 10
    model.parameters["phi"].lower_bound = 0
    model.parameters["phi"].upper_bound = 2
    model.parameters["x0"].lower_bound = 0
    model.parameters["x0"].upper_bound = 1

    y_model = model.func(x, params)

    a_fit = np.zeros(num_datasets)
    a_err = np.zeros_like(a_fit)

    for sample in range(num_datasets):
        y = stats.binom.rvs(n=num_trials, p=y_model, size=y_model.size)
        y = y / num_trials

        fit = BinomialFitter(x=x, y=y, num_trials=num_trials, model=model)

        a_fit[sample] = fit.derived_values["a"]
        a_err[sample] = fit.derived_uncertainties["a"]

    a_nominal = (params["max"] - params["min"]) / 2.0
    a_fit_mean = np.mean(a_fit)
    a_fit_err = np.abs(np.mean(a_fit) - a_nominal)
    a_std_err = np.mean(a_err)
    a_fit_std = np.std(a_fit)

    def plot_fits():
        if not plot_failures:
            return

        num_bins = 100
        _, a_edges = np.histogram(a_fit, num_bins)
        a_bin_centres = (a_edges[:-1] + a_edges[1:]) / 2

        hist_results = plt.hist(a_fit / 0.5, bins=a_edges / 0.5, density=True)
        a_hist = hist_results[0]

        plt.axvline(x=a_nominal / 0.5, color="black", label="nominal")
        plt.axvline(x=(a_nominal + a_fit_std) / 0.5, color="black", linestyle="--")
        plt.axvline(x=(a_nominal - a_fit_std) / 0.5, color="black", linestyle="--")

        plt.axvline(x=a_fit_mean / 0.5, color="blue", label="fitted")
        plt.axvline(x=(a_fit_mean + a_std_err) / 0.5, color="blue", linestyle="--")
        plt.axvline(x=(a_fit_mean - a_std_err) / 0.5, color="blue", linestyle="--")

        hist_model = Gaussian()
        hist_fit = NormalFitter(x=a_bin_centres, y=a_hist, model=hist_model)
        norm_x, norm_y = hist_fit.evaluate()
        plt.plot(norm_x / 0.5, norm_y)

        plt.xlabel("contrast")
        plt.ylabel("relative frequency")
        plt.grid()
        plt.legend()
        plt.show()

    if np.mean(a_fit) - a_nominal > 1e-3:
        plot_fits()
        raise ValueError(f"Error in fitted parameter value too high ({a_fit_err:.3e})")
    if np.abs(1 - a_std_err / a_fit_std) > 0.25:
        plot_fits()
        raise ValueError(
            "Standard error estimate does not match standard deviation of fitted "
            f"parameter values: (standard errors {a_std_err:.3e}, {a_fit_std:.3e})"
        )
