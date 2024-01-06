from matplotlib import pyplot as plt
import numpy as np
import pprint

import ionics_fits as fits
from . import common


def gaussian(x, y, a, x0_x, x0_y, sigma_x, sigma_y, z0):

    A = a / (sigma_x * np.sqrt(2 * np.pi)) / (sigma_y * np.sqrt(2 * np.pi))

    return (
        A
        * np.exp(
            -(((x - x0_x) / (np.sqrt(2) * sigma_x)) ** 2)
            - (((y - x0_y) / (np.sqrt(2) * sigma_y)) ** 2)
        )
        + z0
    )


def parabola(x, y, k_x, k_y, x0, y0, z0):
    return k_x * (x - x0) ** 2 + k_y * (y - y0) ** 2 + z0


def cone(x, y, x0_x, x0_y, k_x, k_y, y0):
    return np.sqrt((k_x * (x - x0_x)) ** 2 + (k_y * (y - x0_y)) ** 2) + y0


def check_param_values(x_mesh_0, x_mesh_1, test_params, fit, func, plot_failures):
    if not common.params_close(test_params, fit.values, 1e-3):
        if plot_failures:

            fig, axs = plt.subplots(2, 1)

            plt.axes(axs[0])

            plt.pcolormesh(x_mesh_0, x_mesh_1, func(x_mesh_0, x_mesh_1, **test_params))
            axs[0].set_title("model")
            plt.grid()
            plt.xlabel("x")
            plt.ylabel("y")

            plt.axes(axs[1])

            plt.pcolormesh(x_mesh_0, x_mesh_1, func(x_mesh_0, x_mesh_1, **fit.values))
            axs[1].set_title("fit")
            plt.grid()
            plt.xlabel("x")
            plt.ylabel("y")

            plt.show()

        params_str = pprint.pformat(test_params, indent=4)
        fitted_params_str = pprint.pformat(fit.values, indent=4)
        initial_params_str = pprint.pformat(fit.initial_values, indent=4)

        raise ValueError(
            "Error in parameter values is too large:\n"
            f"test parameter set was: {params_str}\n"
            f"fitted parameters were: {fitted_params_str}\n"
            f"estimated parameters were: {initial_params_str}\n"
            f"free parameters were: {fit.free_parameters}"
        )


def test_call_2d(plot_failures):
    """Check that the 2D model call / func methods produce the correct output"""
    params = {
        "a": 5,
        "x0_x": -2,
        "x0_y": +0.5,
        "sigma_x": 2,
        "sigma_y": 5,
        "z0": 1.5,
    }

    x_ax_0 = np.linspace(-20, 20, 30)
    x_ax_1 = np.linspace(-50, 50, 70)
    x_mesh_0, x_mesh_1 = np.meshgrid(x_ax_0, x_ax_1)

    y = gaussian(x_mesh_0, x_mesh_1, **params)
    model = fits.multi_x.Gaussian2D()
    y_model = model((x_ax_0, x_ax_1), **params)

    assert common.is_close(y_model.squeeze(), y.T, tol=1e-3)


def test_gaussian_2d(plot_failures):
    """Test 2D Gaussian fitting"""
    params = {
        "a": 5,
        "x0_x": -2,
        "x0_y": +0.5,
        "sigma_x": 2,
        "sigma_y": 5,
        "z0": 1.5,
    }

    x_ax_0 = np.linspace(-20, 20, 30)
    x_ax_1 = np.linspace(-50, 50, 70)
    x_mesh_0, x_mesh_1 = np.meshgrid(x_ax_0, x_ax_1)

    y = gaussian(x_mesh_0, x_mesh_1, **params)

    fit = fits.multi_x.common.Fitter2D(
        x=(x_ax_0, x_ax_1), y=y.T, model=fits.multi_x.Gaussian2D()
    )

    assert set(fit.values.keys()) == set(params.keys())
    assert set(fit.uncertainties.keys()) == set(params.keys())
    assert set(fit.initial_values.keys()) == set(params.keys())
    assert set(fit.free_parameters) == set(params.keys())
    assert set(fit.derived_values.keys()) == set(
        ["FWHMH_x", "w0_x", "FWHMH_y", "peak_y", "w0_y"]
    )
    assert set(fit.derived_uncertainties.keys()) == set(
        ["FWHMH_x", "w0_x", "FWHMH_y", "peak_y", "w0_y"]
    )

    _, y_fit = fit.evaluate()
    assert common.is_close(y_fit.squeeze(), y.T, tol=1e-3)

    residuals = fit.residuals()
    assert common.is_close(residuals.squeeze(), np.zeros_like(y).T, tol=1e-3)

    check_param_values(x_mesh_0, x_mesh_1, params, fit, gaussian, plot_failures)


def test_parabola_2d(plot_failures):
    """Test 2D Parabola fitting"""
    params = {
        "x0": -2,
        "y0": +0.5,
        "k_x": 2,
        "k_y": 5,
        "z0": 1.5,
    }

    x_ax_0 = np.linspace(-20, 20, 30)
    x_ax_1 = np.linspace(-50, 50, 70)
    x_mesh_0, x_mesh_1 = np.meshgrid(x_ax_0, x_ax_1)

    y = parabola(x_mesh_0, x_mesh_1, **params)

    fit = fits.multi_x.common.Fitter2D(
        x=(x_ax_0, x_ax_1), y=y.T, model=fits.multi_x.Parabola2D()
    )

    assert set(fit.values.keys()) == set(params.keys())
    assert set(fit.uncertainties.keys()) == set(params.keys())
    assert set(fit.initial_values.keys()) == set(params.keys())
    assert set(fit.free_parameters) == set(params.keys())
    assert set(fit.derived_values.keys()) == set([])

    _, y_fit = fit.evaluate()
    assert common.is_close(y_fit.squeeze(), y.T, tol=1e-3)

    residuals = fit.residuals()
    assert common.is_close(residuals.squeeze(), np.zeros_like(y).T, tol=1e-3)

    check_param_values(x_mesh_0, x_mesh_1, params, fit, parabola, plot_failures)


def test_cone_2d(plot_failures):
    """Test 2D cone fitting"""
    params = {
        "x0_x": -5,
        "x0_y": +10,
        "k_x": 2,
        "k_y": 5,
        "y0": 0,
    }

    x_ax_0 = np.linspace(-40, 40, 30)
    x_ax_1 = np.linspace(-50, 50, 70)
    x_mesh_0, x_mesh_1 = np.meshgrid(x_ax_0, x_ax_1)

    y = cone(x_mesh_0, x_mesh_1, **params)

    fit = fits.multi_x.common.Fitter2D(
        x=(x_ax_0, x_ax_1),
        y=y.T,
        model=fits.multi_x.models.Cone2D(),
    )

    check_param_values(x_mesh_0, x_mesh_1, params, fit, cone, plot_failures)


def test_laser_flop_2d(plot_failures):
    """Test / example of constructing a 2D fit using the laser flopping model.

    We simulate Rabi flopping on a blue sideband as a function of the alignment between
    the laser and the motional mode.
    """
    t_pi = 5e-6
    omega = np.pi / t_pi
    eta = 0.1
    theta_0 = 0.25

    angle_axis = np.linspace(-np.pi / 2, +np.pi / 3, 50)
    time_axis = np.linspace(0, 5 * (t_pi / eta), 75)
    time_mesh, angle_mesh = np.meshgrid(time_axis, angle_axis)

    flop_model = fits.models.LaserFlopTimeThermal(
        start_excited=False, sideband_index=+1, n_max=1
    )
    flop_model.parameters["n_bar"].fixed_to = 0
    flop_model.parameters["delta"].fixed_to = 0
    flop_model.parameters["omega"].fixed_to = omega
    flop_model.parameters["P_readout_e"].fixed_to = 1
    flop_model.parameters["P_readout_g"].fixed_to = 0

    sinusoid_model = fits.models.Sinusoid()
    sinusoid_model.parameters["omega"].fixed_to = 1
    sinusoid_model.parameters["x0"].fixed_to = -np.pi / 2
    sinusoid_model.parameters["y0"].fixed_to = 0
    sinusoid_model.parameters["phi"].offset = 0

    # Generate data to fit
    y = np.zeros_like(time_mesh)
    for idx, angle in np.ndenumerate(angle_axis):
        eta_angle = float(sinusoid_model(x=angle, a=eta, phi=theta_0))
        y[idx, :] = flop_model(x=time_axis, eta=eta_angle, omega=omega)

    model = fits.multi_x.Model2D(
        models=(flop_model, sinusoid_model),
        result_params=("eta",),
    )

    params = {"omega_x0": omega, "x0_x1": -np.pi / 2}
    fit = fits.multi_x.common.Fitter2D(x=(time_axis, angle_axis), y=y.T, model=model)

    def func(x, y, **kwargs):
        return model.__call__((x, y), **kwargs)

    check_param_values(time_mesh, angle_mesh, params, fit, func, plot_failures)
