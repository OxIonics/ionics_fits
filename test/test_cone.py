import numpy as np

from ionics_fits.models.cone import ConeSlice

from .common import Config, check_multiple_param_sets

x = np.linspace(-2, 2, 100)
params = {
    "x0": -0.25,
    "z0": -1,
    "alpha": +3,
    "k": [-2.7, +3],
}


def test_cone(plot_failures: bool):
    """Test for cone.ConeSlice with neither z0 nor alpha floated"""

    model = ConeSlice()
    model.parameters["z0"].fixed_to = params["z0"]
    model.parameters["alpha"].fixed_to = params["alpha"]

    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )


def test_cone_alpha(plot_failures: bool):
    """Test for cone.ConeSlice with alpha but not z0 floated"""

    model = ConeSlice()
    model.parameters["z0"].fixed_to = params["z0"]

    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )


def test_cone_z0(plot_failures: bool):
    """Test for cone.ConeSlice with z0 but not alpha floated"""

    model = ConeSlice()
    model.parameters["z0"].fixed_to = None
    model.parameters["alpha"].fixed_to = params["alpha"]

    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )


def test_cone_z0_alpha(plot_failures: bool):
    """Test for cone.ConeSlice with both z0 and alpha floated"""

    model = ConeSlice()
    model.parameters["z0"].fixed_to = None
    model.parameters["alpha"].fixed_to = None

    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=None, residual_tol=1e-3),
    )
