import numpy as np

from ionics_fits.binomial import BinomialFitter
from ionics_fits.models.sinusoid import SineMinMax
from ionics_fits.validators import NSigmaValidator


def test_n_sigma_validator():
    params = {
        "min": 0,
        "max": 1,
        "omega": 1 * 2 * np.pi,
        "phi": 0,
        "y0": 0,
    }

    model = SineMinMax()

    x = np.linspace(1, 10)
    y = model(x, **params)

    # Case 1: something we can fit perfectly
    model.parameters["min"].lower_bound = 0
    model.parameters["min"].upper_bound = 1
    model.parameters["max"].lower_bound = 0
    model.parameters["max"].upper_bound = 1

    fit = BinomialFitter(x=x, y=y, model=model, num_trials=100)

    validator = NSigmaValidator()
    success, p = validator.validate(fit)

    assert success
    assert p == 1.0

    # Case 1: something we can't fit
    model.parameters["min"].lower_bound = 0.2
    model.parameters["min"].upper_bound = 0.5
    model.parameters["max"].lower_bound = 0.5
    model.parameters["max"].upper_bound = 0.8

    fit = BinomialFitter(x=x, y=y, model=model, num_trials=100)

    validator = NSigmaValidator()
    success, p = validator.validate(fit)

    assert not success
    assert p < 0.5
