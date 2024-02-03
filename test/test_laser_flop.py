import numpy as np

from ionics_fits.models.laser_rabi import (
    LaserFlopFreqCoherent,
    LaserFlopFreqSqueezed,
    LaserFlopFreqThermal,
    LaserFlopTimeCoherent,
    LaserFlopTimeThermal,
    LaserFlopTimeSqueezed,
)
from .common import check_multiple_param_sets, Config


def _test_laser_flop_freq(
    plot_failures: bool,
    P_readout_e: float,
    sideband_index: int,
    flop_class,
    dist_params,
):
    t_pi = 5e-6
    params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "w_0": 0.5e6 * 2 * np.pi,
        "omega": np.pi / t_pi,
        "t_dead": 0.0,
        "eta": 0.1,
        "tau": np.inf,
    }
    params["t_pulse"] = 1 * t_pi
    params["t_pulse"] /= params["eta"] ** abs(sideband_index)
    params.update(dist_params)

    w = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    w *= params["eta"] ** abs(sideband_index)

    model = flop_class(start_excited=True, sideband_index=sideband_index, n_max=100)
    model.parameters["P_readout_e"].fixed_to = params["P_readout_e"]
    model.parameters["P_readout_g"].fixed_to = params["P_readout_g"]
    model.parameters["eta"].fixed_to = params["eta"]
    model.parameters["t_pulse"].fixed_to = params["t_pulse"]

    # It's hard for the fit to converge when both omega and n_bar are floated since
    # there is a lot of covariance between these parameters. Here we test the case where
    # omega is fixed and n_bar is floated
    model.parameters["omega"].fixed_to = params["omega"]

    check_multiple_param_sets(
        w,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=None, residual_tol=1e-4),
    )


def test_laser_flop_freq(plot_failures: bool):
    """Test for laser_rabi.LaserFlopFreq"""
    _test_laser_flop_freq(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=-1,
        flop_class=LaserFlopFreqCoherent,
        dist_params={"alpha": np.sqrt(4)},
    )
    _test_laser_flop_freq(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=0,
        flop_class=LaserFlopFreqThermal,
        dist_params={"n_bar": [0, 1, 5, 10]},
    )
    _test_laser_flop_freq(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=-1,
        flop_class=LaserFlopFreqThermal,
        dist_params={"n_bar": [0.1, 1]},
    )
    _test_laser_flop_freq(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=+1,
        flop_class=LaserFlopFreqThermal,
        dist_params={"n_bar": [0.1, 0.1, 1]},
    )
    _test_laser_flop_freq(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=+1,
        flop_class=LaserFlopFreqSqueezed,
        dist_params={"zeta": 1.0},
    )


def _test_laser_flop_time(
    plot_failures: bool,
    P_readout_e: float,
    sideband_index: int,
    flop_class,
    dist_params,
):
    t_pi = 5e-6
    params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "delta": 2 * np.pi * 0,
        "omega": np.pi / t_pi,
        "t_dead": 0.0,
        "eta": 0.1,
        "tau": np.inf,
    }
    params.update(dist_params)

    t_pulse = np.linspace(0, 5 * t_pi, 100)
    t_pulse /= params["eta"] ** abs(sideband_index)

    model = flop_class(start_excited=True, sideband_index=sideband_index, n_max=100)
    model.parameters["P_readout_e"].fixed_to = params["P_readout_e"]
    model.parameters["P_readout_g"].fixed_to = params["P_readout_g"]
    model.parameters["eta"].fixed_to = params["eta"]

    # It's hard for the fit to converge when both omega and n_bar are floated since
    # there is a lot of covariance between these parameters. Here we test the case where
    # omega is fixed and n_bar is floated
    model.parameters["omega"].fixed_to = params["omega"]

    check_multiple_param_sets(
        t_pulse,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=None, residual_tol=1e-4),
    )


def test_laser_flop_time(plot_failures: bool):
    """Test for laser_rabi.LaserFlopTime"""
    _test_laser_flop_time(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=-1,
        flop_class=LaserFlopTimeCoherent,
        dist_params={"alpha": np.sqrt(4)},
    )
    _test_laser_flop_time(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=0,
        flop_class=LaserFlopTimeThermal,
        dist_params={"n_bar": [0, 1, 5, 10]},
    )
    _test_laser_flop_time(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=-1,
        flop_class=LaserFlopTimeThermal,
        dist_params={"n_bar": [0, 0.1, 1]},
    )
    _test_laser_flop_time(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=+1,
        flop_class=LaserFlopTimeThermal,
        dist_params={"n_bar": [0, 0.1, 1]},
    )
    _test_laser_flop_time(
        plot_failures=plot_failures,
        P_readout_e=1.0,
        sideband_index=+1,
        flop_class=LaserFlopTimeSqueezed,
        dist_params={"zeta": 1.0},
    )
