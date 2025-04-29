"""Tests for fitting models with multiple y-axis dimensions"""

import numpy as np

from ionics_fits.models.rabi import RabiFlopFreq
from ionics_fits.models.transformations.repeated_model import RepeatedModel

from .common import Config, check_multiple_param_sets


class DoubleRabiFreq(RepeatedModel):
    def __init__(self):
        super().__init__(
            model=RabiFlopFreq(start_excited=True),
            common_params=[
                "P_readout_e",
                "P_readout_g",
                "t_pulse",
                "omega",
                "tau",
                "t_dead",
            ],
            num_repetitions=2,
        )


def test_multi_y(plot_failures):
    """Test fitting to a model with multiple y-axis dimensions"""
    w = np.linspace(-10, 10, 200)
    params = {
        "w_0_0": [-3.0],
        "w_0_1": [+3.0],
        "P_readout_e": 1,
        "P_readout_g": 0,
        "t_pulse": 1,
        "omega": np.pi,
        "tau": np.inf,
        "t_dead": 0,
    }

    model = DoubleRabiFreq()
    model.parameters["P_readout_e"].fixed_to = params["P_readout_e"]
    model.parameters["P_readout_g"].fixed_to = params["P_readout_g"]
    model.parameters["t_pulse"].fixed_to = params["t_pulse"]

    check_multiple_param_sets(
        w,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=None, residual_tol=1e-4),
    )
