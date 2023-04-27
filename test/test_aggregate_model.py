from matplotlib import pyplot as plt

import numpy as np

import ionics_fits as fits
from . import common


def test_aggregate_model_func(plot_failures):
    """Test that aggregate_model.AggregateModel.func returns the correct value"""
    x = np.linspace(-10, 10, 100)
    line = fits.models.Line()
    triangle = fits.models.Triangle()
    model = fits.models.AggregateModel(models=[("line", line), ("triangle", triangle)])

    params = {
        "line_y0": 3,
        "line_a": 0.5,
        "triangle_x0": 1,
        "triangle_y0": 1,
        "triangle_k": 2.5,
        "triangle_sym": 0,
        "triangle_y_min": -np.inf,
        "triangle_y_max": +np.inf,
    }

    y_line = line.func(x, {"a": params["line_a"], "y0": params["line_y0"]})
    y_triangle = triangle.func(
        x,
        {
            param_name: params[f"triangle_{param_name}"]
            for param_name in triangle.parameters.keys()
        },
    )

    y_aggregate = model.func(x, params)

    success = np.abs(np.max(np.stack((y_line, y_triangle)).T - y_aggregate)) < 1e-10

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
    line = fits.models.Line()
    triangle = fits.models.Triangle()
    model = fits.models.AggregateModel(models=[("line", line), ("triangle", triangle)])

    params = {
        "line_y0": [3],
        "line_a": [0.5],
        "triangle_x0": [1],
        "triangle_y0": [1],
        "triangle_k": [2.5],
        "triangle_sym": [0],
        "triangle_y_min": [-np.inf],
        "triangle_y_max": [+np.inf],
    }

    common.check_multiple_param_sets(
        x,
        model,
        params,
        common.TestConfig(plot_failures=plot_failures),
    )
