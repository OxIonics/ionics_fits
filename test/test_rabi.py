from typing import Optional
import numpy as np

import ionics_fits as fits
from . import common


def _test_rabi_freq(P_readout_e: float):
    """Test for rabi.RabiFlopFreq"""
    w = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    t_pulse = 5e-6
    params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "w_0": 0.5e6 * 2 * np.pi,
        "omega": np.array([0.1, 0.25, 1]) * np.pi / t_pulse,
        "t_pulse": t_pulse,
        "t_dead": 0.0,
        "tau": np.inf,
    }

    common.check_multiple_param_sets(
        w,
        fits.models.RabiFlopFreq(start_excited=True),
        params,
        common.TestConfig(plot_failures=True, param_tol=None, residual_tol=1e-4),
    )


def test_rabi_freq():
    """Test for rabi.RabiFlopFreq"""
    _test_rabi_freq(P_readout_e=1.0)


def test_rabi_freq_w0_only():
    """Test for rabi.RabiFlopFreq in the case where w_0 is the only unknown.

    This case is special-cased in the parameter estimator.
    """
    w = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    t_pulse = 5e-6
    params = {
        "P_readout_e": 1,
        "P_readout_g": 0,
        "w_0": [0, -0.25e6 * 2 * np.pi, 0.5e6 * 2 * np.pi],
        "t_pulse": t_pulse,
        "t_dead": 0.0,
        "tau": np.inf,
    }

    for omega in np.array([0.1, 0.25, 1]):
        params["omega"] = omega * np.pi / t_pulse
        model = fits.models.RabiFlopFreq(start_excited=True)
        model.parameters["omega"].fixed_to = params["omega"]
        model.parameters["t_pulse"].fixed_to = params["t_pulse"]
        common.check_multiple_param_sets(
            w,
            model,
            params,
            common.TestConfig(plot_failures=True, param_tol=None, residual_tol=1e-4),
        )


def test_rabi_freq_inverted():
    """Test for rabi.RabiFlopFreq, with the readout levels inverted"""
    _test_rabi_freq(P_readout_e=0.0)


def _test_rabi_time(P_readout_e: float):
    t_pulse = np.linspace(0, 20e-6, 100) * 2 * np.pi
    params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "delta": 2 * np.pi * np.array([0, 0.025e6]),
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
        "t_dead": 0.0,
        "tau": np.inf,
    }

    common.check_multiple_param_sets(
        t_pulse,
        fits.models.RabiFlopTime(start_excited=True),
        params,
        common.TestConfig(plot_failures=True, param_tol=None, residual_tol=1e-4),
    )


def test_rabi_time():
    """Test for rabi.RabiFlopTime"""
    _test_rabi_time(P_readout_e=1.0)


def test_rabi_time_inverted():
    """Test for rabi.RabiFlopTime, with the readout levels inverted"""
    _test_rabi_time(P_readout_e=0.0)


def _fuzz_rabi_freq(
    P_readout_e: float,
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    delta = np.linspace(-2e6, 2e6, 200) * 2 * np.pi
    fuzzed_params = {
        "w_0": 2 * np.pi * np.array([-0.5e6, 0.5e6]),
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
        "t_pulse": (3e-6, 5e-6),
    }

    static_params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "t_dead": 0.0,
        "tau": np.inf,
    }
    model = fits.models.RabiFlopFreq(start_excited=True)

    test_config = test_config or common.TestConfig()
    test_config.plot_failures = True

    # Parameter tolerances are messy here since the tolerance for delta needs to be
    # judged relative to the size of Omega, which the code isn't set up for. So we check
    # residuals rather than values of individual parameters.
    #
    # More generally: we're floating both omega and delta in these fits, which leads to
    # a parameter space with a lot of local minima. As a result, we have to set
    # realistic expectations for the fit accuracy - to always get us within the basin of
    # the global minimum, the heuristics need to be unreasonably accurate. If we just
    # floated one of Omega / delta in the fits we could (should) set a much tighter
    # tolerance here...
    test_config.param_tol = None
    test_config.residual_tol = 1e-2

    return common.fuzz(
        x=delta,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )


def _fuzz_rabi_time(
    P_readout_e: float,
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    t = np.linspace(0, 20e-6, 400) * 2 * np.pi
    fuzzed_params = {
        "delta": (0, 25e3 * 2 * np.pi),
        "omega": np.pi / 5e-6 * np.array([0.25, 1.0]),
    }

    static_params = {
        "P_readout_e": P_readout_e,
        "P_readout_g": 1 - P_readout_e,
        "t_dead": 0.0,
        "tau": np.inf,
    }

    model = fits.models.RabiFlopTime(start_excited=True)

    # If we float the readout levels and the detuning the fits are under-defined
    model.parameters["P_readout_e"].fixed_to = P_readout_e
    model.parameters["P_readout_g"].fixed_to = 1 - P_readout_e

    test_config = test_config or common.TestConfig()
    test_config.plot_failures = True

    # Parameter tolerances are messy here since the tolerance for delta needs to be
    # judged relative to the size of Omega, which the code isn't set up for. So we check
    # residuals rather than values of individual parameters.
    #
    # More generally: we're floating both omega and delta in these fits, which leads to
    # a parameter space with a lot of local minima. As a result, we have to set
    # realistic expectations for the fit accuracy - to always get us within the basin of
    # the global minimum, the heuristics need to be unreasonably accurate. If we just
    # floated one of Omega / delta in the fits we could (should) set a much tighter
    # tolerance here...
    test_config.param_tol = None
    test_config.residual_tol = 1e-2

    return common.fuzz(
        x=t,
        model=model,
        static_params=static_params,
        fuzzed_params=fuzzed_params,
        test_config=test_config,
        fitter_cls=None,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
    )


def fuzz_rabi_time(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    return _fuzz_rabi_time(
        P_readout_e=1.0,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )


def fuzz_rabi_time_inverted(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    return _fuzz_rabi_time(
        P_readout_e=0.0,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )


def fuzz_rabi_freq(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    return _fuzz_rabi_freq(
        P_readout_e=1.0,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )


def fuzz_rabi_freq_inverted(
    num_trials: int = 100,
    stop_at_failure: bool = True,
    test_config: Optional[common.TestConfig] = None,
) -> float:
    return _fuzz_rabi_freq(
        P_readout_e=0.0,
        num_trials=num_trials,
        stop_at_failure=stop_at_failure,
        test_config=test_config,
    )
