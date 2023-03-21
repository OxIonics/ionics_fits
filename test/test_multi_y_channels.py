""" Tests for fitting models with multiple y channels """
from typing import Dict, TYPE_CHECKING
import numpy as np

import ionics_fits as fits
from . import common


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float


class DoubleRabiFreq(fits.Model):
    def __init__(self):
        parameters = {
            "w_0_0": fits.ModelParameter(scale_func=lambda x_scale, y_scale, _: None),
            "w_0_1": fits.ModelParameter(scale_func=lambda x_scale, y_scale, _: None),
        }
        super().__init__(parameters)

        self.model = fits.models.RabiFlopFreq(start_excited=True)
        self.rabi_params = {
            "P_readout_e": 1,
            "P_readout_g": 0,
            "t_pulse": 1,
            "omega": np.pi,
            "tau": np.inf,
            "t_dead": 0,
        }
        for param, value in self.rabi_params.items():
            self.model.parameters[param].fixed_to = value

    def get_num_y_channels(self) -> int:
        """Returns the number of y channels supported by the model"""
        return 2

    def func(
        self,
        x: fits.utils.Array[("num_samples",), np.float64],
        param_values: Dict[str, float],
    ) -> fits.utils.Array[("num_samples", "num_y_channels"), np.float64]:
        """
        Return measurement probability as function of pulse frequency.

        :param x: Angular frequency
        """
        params = self.rabi_params

        params["w_0"] = param_values["w_0_0"]
        y_0 = self.model.func(x, params)

        params["w_0"] = param_values["w_0_1"]
        y_1 = self.model.func(x, params)

        return np.stack((y_0, y_1)).T

    def estimate_parameters(
        self,
        x: fits.utils.Array[("num_samples",), np.float64],
        y: fits.utils.Array[("num_samples", "num_y_channels"), np.float64],
        model_parameters: Dict[str, fits.ModelParameter],
    ):
        """Set heuristic values for model parameters.

        Typically called during `Fitter.fit`. This method may make use of information
        supplied by the user for some parameters (via the `fixed_to` or
        `user_estimate` attributes) to find initial guesses for other parameters.

        The datasets must be sorted in order of increasing x-axis values and must not
        contain any infinite or nan values. If all parameters of the model allow
        rescaling, then `x`, `y` and `model_parameters` will contain rescaled values.

        :param x: x-axis data, rescaled if allowed.
        :param y: y-axis data, rescaled if allowed.
        :param model_parameters: dictionary mapping model parameter names to their
            metadata, rescaled if allowed.
        """
        self.model.parameters["w_0"].heuristic = None
        self.model.estimate_parameters(x, y[:, 0], self.model.parameters)
        model_parameters["w_0_0"].heuristic = self.model.parameters["w_0"].heuristic

        self.model.parameters["w_0"].heuristic = None
        self.model.estimate_parameters(x, y[:, 1], self.model.parameters)
        model_parameters["w_0_1"].heuristic = self.model.parameters["w_0"].heuristic


def test_multi_y():
    """Test fitting to a model with multiple y channels"""
    w = np.linspace(-10, 10, 200)
    params = {
        "w_0_0": [-3.0],
        "w_0_1": [+3.0],
    }

    common.check_multiple_param_sets(
        w,
        DoubleRabiFreq(),
        params,
        common.TestConfig(plot_failures=True, param_tol=None, residual_tol=1e-4),
    )
