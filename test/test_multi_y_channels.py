""" Tests for fitting models with multiple y channels """
from typing import TYPE_CHECKING
import numpy as np

import ionics_fits as fits
from . import common


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class DoubleRabiFreq(fits.models.RepeatedModel):
    def __init__(self):
        super().__init__(
            model=fits.models.RabiFlopFreq(start_excited=True),
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
    """Test fitting to a model with multiple y channels"""
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

    common.check_multiple_param_sets(
        w,
        model,
        params,
        common.TestConfig(
            plot_failures=plot_failures, param_tol=None, residual_tol=1e-4
        ),
    )
