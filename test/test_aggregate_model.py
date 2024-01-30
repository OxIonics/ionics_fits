from matplotlib import pyplot as plt

import numpy as np

from ionics_fits.models.laser_rabi import LaserFlopTimeThermal
from ionics_fits.models.polynomial import Line
from ionics_fits.models.triangle import Triangle
from ionics_fits.models.transformations.aggregate_model import AggregateModel
from .common import check_multiple_param_sets, Config


def test_aggregate_model_common_params(plot_failures):
    """Test for aggregate_model.AggregateModel with common parameters"""

    rsb = LaserFlopTimeThermal(start_excited=False, sideband_index=-1)
    bsb = LaserFlopTimeThermal(start_excited=False, sideband_index=+1)

    model = AggregateModel(
        models={"rsb": rsb, "bsb": bsb},
        common_params={
            param: (rsb.parameters[param], [("rsb", param), ("bsb", param)])
            for param in rsb.parameters.keys()
        },
    )

    t_pi = 10e-6
    t = np.linspace(0, 3, 20) * t_pi
    eta = 0.1
    params = {
        "P_readout_e": 1,
        "P_readout_g": 0,
        "eta": eta,
        "omega": np.pi / (eta * t_pi),
        "delta": 0,
    }
    for param in params:
        model.parameters[param].fixed_to = params[param]

    params.update(
        {
            param_name: model.parameters[param_name].fixed_to
            for param_name in model.parameters.keys()
            if model.parameters[param_name].fixed_to is not None
        }
    )

    params["n_bar"] = [0, 0.25, 1]
    check_multiple_param_sets(
        x=t,
        model=model,
        test_params=params,
        config=Config(plot_failures=plot_failures),
    )


def test_aggregate_model_func(plot_failures):
    """Test that aggregate_model.AggregateModel.func returns the correct value"""
    x = np.linspace(-10, 10, 100)
    line = Line()
    triangle = Triangle()
    model = AggregateModel(models={"line": line, "triangle": triangle})

    params = {
        "y0_line": 3,
        "a_line": 0.5,
        "x0_triangle": 1,
        "y0_triangle": 1,
        "k_triangle": 2.5,
        "sym_triangle": 0,
        "y_min_triangle": -np.inf,
        "y_max_triangle": +np.inf,
    }

    y_line = line.func(x, {"a": params["a_line"], "y0": params["y0_line"]})
    y_triangle = triangle.func(
        x,
        {
            param_name: params[f"{param_name}_triangle"]
            for param_name in triangle.parameters.keys()
        },
    )

    y_aggregate = model.func(x, params)

    success = np.abs(np.max(np.vstack((y_line, y_triangle)) - y_aggregate)) < 1e-10

    if plot_failures and not success:
        _, ax = plt.subplots(2, 1)

        ax[0].plot(x, y_line, label="model")
        ax[0].plot(x, y_aggregate[:, 0], "--", label="aggregate")
        ax[0].legend()

        ax[1].plot(x, y_triangle, label="model")
        ax[1].plot(x, y_aggregate[:, 1], "--", label="aggregate")
        ax[1].legend()

        plt.show()

    if not success:
        raise ValueError("Aggregate model evaluation does not match individual models")


def test_aggregate_model(plot_failures):
    """Test for aggregate_model.AggregateModel"""
    x = np.linspace(0, 2, 100)
    line = Line()
    triangle = Triangle()
    model = AggregateModel(models={"line": line, "triangle": triangle})

    params = {
        "y0_line": [3],
        "a_line": [0.5],
        "x0_triangle": [1],
        "y0_triangle": [1],
        "k_triangle": [2.5],
        "sym_triangle": [0],
        "y_min_triangle": [-np.inf],
        "y_max_triangle": [+np.inf],
    }

    check_multiple_param_sets(
        x,
        model,
        params,
        Config(plot_failures=plot_failures),
    )
