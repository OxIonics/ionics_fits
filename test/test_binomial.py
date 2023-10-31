import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

import ionics_fits as fits
from . import common


def test_binomial(plot_failures):
    """Basic test of binomial fitting"""
    num_trials = 1000
    x = np.linspace(-3, 3, 200) * 2 * np.pi
    model = fits.models.Sinusoid()
    params = {
        "a": 0,
        "y0": 0,
        "P_lower": 0,
        "P_upper": 1,
        "omega": 1,
        "phi": 1,
        "x0": 0,
        "tau": np.inf,
    }
    model.parameters["a"].fixed_to = params["a"]
    model.parameters["y0"].fixed_to = params["y0"]
    model.parameters["P_lower"].fixed_to = None
    model.parameters["P_upper"].fixed_to = None

    common.check_single_param_set(
        x=x,
        model=model,
        test_params=params,
        config=common.TestConfig(plot_failures=plot_failures),
        fitter_cls=fits.BinomialFitter,
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
    model = fits.models.Sinusoid()
    contrast = 0.995
    offset = 0.5

    params = {
        "P_lower": offset - 0.5 * contrast,
        "P_upper": offset + 0.5 * contrast,
        "a": 0,
        "y0": 0,
        "omega": 1,
        "phi": 1,
        "x0": 0,
        "tau": np.inf,
    }

    model.parameters["P_lower"].fixed_to = None
    model.parameters["P_upper"].fixed_to = None
    model.parameters["a"].fixed_to = params["a"]
    model.parameters["y0"].fixed_to = params["y0"]
    model.parameters["omega"].fixed_to = params["omega"]
    model.parameters["x0"].fixed_to = params["x0"]
    model.parameters["phi"].fixed_to = params["phi"]

    model.parameters["omega"].lower_bound = 0
    model.parameters["omega"].upper_bound = 10
    model.parameters["phi"].lower_bound = 0
    model.parameters["phi"].upper_bound = 2
    model.parameters["x0"].lower_bound = 0
    model.parameters["x0"].upper_bound = 1

    y_model = model.func(x, params)

    contrast_fit = np.zeros(num_datasets)
    contrast_err = np.zeros_like(contrast_fit)

    for sample in range(num_datasets):
        y = stats.binom.rvs(n=num_trials, p=y_model, size=y_model.size)
        y = y / num_trials

        fit = fits.BinomialFitter(x=x, y=y, num_trials=num_trials, model=model)

        contrast_fit[sample] = fit.derived_values["contrast"]
        contrast_err[sample] = fit.derived_uncertainties["contrast"]

    contrast_fit_mean = np.mean(contrast_fit)
    contrast_fit_err = np.abs(np.mean(contrast_fit) - contrast)
    contrast_std_err = np.mean(contrast_err)
    contrast_fit_std = np.std(contrast_fit)

    def plot_fits():
        if not plot_failures:
            return

        num_bins = 100
        _, contrast_edges = np.histogram(contrast_fit, num_bins)
        contrast_bin_centres = (contrast_edges[:-1] + contrast_edges[1:]) / 2

        hist_results = plt.hist(contrast_fit, bins=contrast_edges, density=True)
        contrast_hist = hist_results[0]

        plt.axvline(x=contrast, color="black", label="nominal")
        plt.axvline(x=contrast + contrast_fit_std, color="black", linestyle="--")
        plt.axvline(x=contrast - contrast_fit_std, color="black", linestyle="--")

        plt.axvline(x=contrast_fit_mean, color="blue", label="fitted")
        plt.axvline(
            x=contrast_fit_mean + contrast_std_err, color="blue", linestyle="--"
        )
        plt.axvline(
            x=contrast_fit_mean - contrast_std_err, color="blue", linestyle="--"
        )

        hist_model = fits.models.Gaussian()
        hist_fit = fits.NormalFitter(
            x=contrast_bin_centres, y=contrast_hist, model=hist_model
        )
        plt.plot(*hist_fit.evaluate(True))

        plt.xlabel("contrast")
        plt.ylabel("relative frequency")
        plt.grid()
        plt.legend()
        plt.show()

    print(np.mean(contrast_fit), contrast)
    print(contrast_std_err, contrast_fit_std)
    print(contrast_err)

    if np.mean(contrast_fit) - contrast > 1e-3:
        plot_fits()
        raise ValueError(
            f"Error in fitted parameter value too high ({contrast_fit_err:.3e})"
        )
    if np.abs(1 - contrast_std_err / contrast_fit_std) > 0.25:
        plot_fits()
        raise ValueError(
            "Standard error estimate does not match standard deviation of fitted "
            f"parameter values: (standard errors {contrast_std_err:.3e}, "
            f"{contrast_fit_std:.3e})"
        )
