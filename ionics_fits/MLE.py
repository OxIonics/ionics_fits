import logging
import numpy as np
import pprint
from scipy import optimize
from typing import Callable, Dict, Tuple, TYPE_CHECKING

from . import Fitter, ModelParameter
from .utils import Array, ArrayLike

if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


logger = logging.getLogger(__name__)


class MLEFitter(Fitter):
    """Base class for maximum Likelihood Parameter Estimation fitters."""

    TYPE: str = "MLE"

    def cost_fun(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_y_channels", "num_samples"), np.float64],
        free_func: Callable[..., Array[("num_y_channels", "num_samples"), np.float64]],
    ) -> float:
        """Returns the negative log-likelihood of a given dataset

        :param free_param_values: array of floated parameter values
        :param x: x-axis data
        :param y: y-axis data
        :param free_func: convenience wrapper for the model function, taking only values
            for the fit's free parameters
        """
        raise NotImplementedError

    def _fit(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_y_channels", "num_samples"), np.float64],
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., Array[("num_y_channels", "num_samples"), np.float64]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the parameter estimation.

        `Fitter` implementations must override this method to provide a fit with
        appropriate statistics.

        :param x: rescaled x-axis data, must be a 1D array
        :param y: rescaled y-axis data
        :param parameters: dictionary of rescaled model parameters
        :param free_func: convenience wrapper for the model function, taking only values
            for the fit's free parameters

        :returns: tuple of dictionaries mapping model parameter names to their fitted
            values and uncertainties.
        """
        free_parameters = [
            param_name
            for param_name, param_data in parameters.items()
            if param_data.fixed_to is None
        ]

        p0 = [parameters[param].get_initial_value() for param in free_parameters]
        lower = [parameters[param].lower_bound for param in free_parameters]
        upper = [parameters[param].upper_bound for param in free_parameters]
        p0_dict = {
            param: parameters[param].get_initial_value() for param in free_parameters
        }

        assert x.dtype == np.float64
        assert y.dtype == np.float64

        logger.debug(
            f"Starting {self.TYPE} fitting with initial parameters: "
            f"{pprint.pformat(p0_dict)}"
        )

        res = optimize.minimize(
            fun=self.cost_fun,
            args=(x, y, free_func),
            x0=p0,
            bounds=zip(lower, upper),
        )

        if not res.success:
            raise RuntimeError(f"{self.TYPE} fit failed: {res.message}")

        p = {param: value for param, value in zip(free_parameters, res.x)}
        p_err = {param: np.nan for param in free_parameters}

        y_fit = free_func(x, *res.x)
        residual_variance = np.sum((y - y_fit) ** 2) / (y.size - len(free_parameters))
        p_cov = res.hess_inv.todense() * residual_variance
        p_err = np.sqrt(np.diag(p_cov))
        p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err
