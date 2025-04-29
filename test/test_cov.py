import numpy as np

from ionics_fits.models.polynomial import Line
from ionics_fits.normal import NormalFitter

from .common import is_close


def test_covariance_scales(plot_failures: bool):
    """Self-consistency check on how we handle rescaling the covariance matrix.

    This test checks that the way we rescale the covariance matrix is consistent with
    our scaling of the uncertainty vector.
    """
    x = np.linspace(-10, 10)
    params = {"a": 3.2, "y0": -9}
    model = Line()
    fit = NormalFitter(x=x, y=model(x, **params), model=model)
    p_err = [fit.uncertainties[param] for param in fit.free_parameters]
    assert is_close(p_err, np.sqrt(fit.covariances), 1e-4)
