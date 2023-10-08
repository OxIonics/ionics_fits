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
        "a": 0.5,
        "omega": 1,
        "phi": 1,
        "y0": 0.5,
        "x0": 0,
        "tau": np.inf,
    }
    model.parameters["a"].fixed_to = params["a"]
    model.parameters["y0"].fixed_to = params["y0"]

    common.check_single_param_set(
        x=x,
        model=model,
        test_params=params,
        config=common.TestConfig(plot_failures=plot_failures),
        fitter_cls=fits.BinomialFitter,
        fitter_args={"num_trials": num_trials},
    )


def test_binomial_synthetic(plot_failures):
    """Test binomial fitting to multiple synthetic datasets"""
    # TODO: set a seed to make this deterministic!
    # TODO: make test quantitative? Fit histogram to a normal dist to check
    #   error bars?
    # TODO: use a more interesting fit model like a sinusoid with readout levels
    num_trials = 100
    x = np.linspace(-3, 3, 200) * 2 * np.pi
    model = fits.models.Sinusoid()
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

    model.parameters["omega"].lower_bound = 0
    model.parameters["omega"].upper_bound = 10
    model.parameters["phi"].lower_bound = 0
    model.parameters["phi"].upper_bound = 2
    model.parameters["x0"].lower_bound = 0
    model.parameters["x0"].upper_bound = 1

    y_model = model.func(x, params)

    num_samples = 1000
    phi_fit = np.zeros(num_samples)

    for sample in range(num_samples):
        y = stats.binom.rvs(n=num_trials, p=y_model, size=y_model.size)
        y = y / num_trials

        fit = fits.BinomialFitter(x=x, y=y, num_trials=num_trials, model=model)

        phi_fit[sample] = fit.values["phi"]

    plt.hist(phi_fit, bins=50)
    plt.axvline(x=1, color="black")
    plt.axvline(x=1 + fit.uncertainties["phi"], color="black", linestyle="--")
    plt.axvline(x=1 - fit.uncertainties["phi"], color="black", linestyle="--")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_binomial_synthetic(True)
