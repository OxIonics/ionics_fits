import numpy as np

from ionics_fits.models.gaussian import Gaussian
from ionics_fits.normal import NormalFitter

from .common import Config, check_multiple_param_sets, fuzz


def test_gaussian(plot_failures):
    """Test for gaussian.Gaussian"""
    x = np.linspace(-12, 20, 500)
    params = {
        "x0": [-3, 1, 0, 0.25, 10],
        "y0": [-1, 0, 1],
        "a": [-5, 5],
        "sigma": [0.1, 0.25, 1],
    }
    model = Gaussian()
    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, heuristic_tol=3e-2),
    )


def test_152(plot_failures):
    """Regression test for issue #152

    This case is challenging for the FFT-based heuristic because there aren't enough
    data points for it to work effectively.
    """
    x = np.array(
        [
            1847.08794244,
            1927.08794244,
            1827.08794244,
            1867.08794244,
            1767.08794244,
            1757.08794244,
            1877.08794244,
            1837.08794244,
            1897.08794244,
            1807.08794244,
            1747.08794244,
            1777.08794244,
        ]
    )

    y = np.array([0.32, 0.94, 0.16, 0.64, 0.26, 0.44, 0.88, 0.26, 1.0, 0.0, 0.5, 0.1])
    sigma = np.array(
        [
            0.0751683,
            0.04370707,
            0.0613972,
            0.07703657,
            0.07132427,
            0.07930124,
            0.05567317,
            0.07132427,
            0.01807572,
            0.01807572,
            0.07980068,
            0.05224949,
        ]
    )

    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    sigma = sigma[inds]

    model = Gaussian()
    model.parameters["sigma"].lower_bound = 1e-9
    fit = NormalFitter(x=x, y=y, sigma=sigma, model=model)

    assert np.abs(fit.initial_values["x0"] - 1800) < 10
    assert np.abs(1 - fit.initial_values["sigma"] / 43.5) < 0.25
    assert np.abs(1 - fit.initial_values["y0"]) < 0.1
    assert np.abs(1 - -fit.initial_values["a"] / 120) < 0.35


def fuzz_gaussian(
    num_trials: int,
    stop_at_failure: bool,
    test_config: Config,
) -> float:
    x = np.linspace(-10, 10, 500)
    fuzzed_params = {
        "x0": (0, 0.25),
        "y0": (-1, 0, 1),
        "a": (-5, 5),
        "sigma": (0.1, 0.25, 1),
    }

    return fuzz(
        x=x,
        model=Gaussian(),
        static_params={},
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        param_generator=None,
    )
