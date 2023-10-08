import numpy as np

import ionics_fits as fits
from . import common


def test_binomial(plot_failures):
    """Test binomial fitting"""
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
    # FIXME!
    model.parameters["a"].scale_func = lambda x_scale, y_scale, _: None

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
