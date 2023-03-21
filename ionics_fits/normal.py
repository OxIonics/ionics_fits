import logging
import numpy as np
import pprint
from scipy import optimize, stats
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

from . import Fitter, ModelParameter
from .utils import Array

if TYPE_CHECKING:
    num_samples = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


logger = logging.getLogger(__name__)


class NormalFitter(Fitter):
    """Fitter for normally-distributed data.

    We use least-squares fitting as a maximum-likelihood parameter estimator for
    normally distributed data. For data that is close to normal this is usually a pretty
    good approximation of a true MLE estimator. YMMV...
    """

    @staticmethod
    def _fit(
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples", "num_y_channels"), np.float64],
        sigma: Optional[Array[("num_samples", "num_y_channels"), np.float64]],
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., Array[("num_samples", "num_y_channels"), np.float64]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Implementation of the parameter estimation.

        `Fitter` implementations must override this method to provide a fit with
        appropriate statistics.

        :param x: rescaled x-axis data
        :param y: rescaled y-axis data
        :param sigma: rescaled standard deviations
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
            "Starting least-squares fitting with initial parameters: "
            f"{pprint.pformat(p0_dict)}"
        )

        def fit_func(
            _: Array[("num_samples_flattened",), np.float64], *free_param_values: float
        ) -> Array[("num_samples_flattened",), np.float64]:
            """Call the model function with the values of the free parameters."""
            return free_func(x, *free_param_values).ravel()

        if y.ndim == 1:
            x_flat = x
            y_flat = y
            sigma_flat = None if sigma is None else sigma
        else:
            x_flat = np.ravel(np.tile(x, (y.ndim, 1)))
            y_flat = np.ravel(y)
            sigma_flat = None if sigma is None else np.ravel(sigma)

        p, p_cov = optimize.curve_fit(
            f=fit_func,
            xdata=x_flat,
            ydata=y_flat,
            p0=p0,
            sigma=sigma_flat,
            absolute_sigma=sigma is not None,
            bounds=(lower, upper),
            method="trf",
        )

        p_err = np.sqrt(np.diag(p_cov))

        p = {param: value for param, value in zip(free_parameters, p)}
        p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err

    def _fit_significance(self) -> Optional[float]:
        """Returns an estimate of the goodness of fit as a number between 0 and 1 or
        `None` if `sigma` has not been supplied.

        This is the defined as the probability that fit residuals as large as the ones
        we observe could have arisen through chance given our assumed statistics and
        assuming that the fitted model perfectly represents the probability distribution

        A value of `1` indicates a perfect fit (all data points lie on the fitted curve)
        a value close to 0 indicates significant deviations of the dataset from the
        fitted model.
        """
        # TODO: support multiple y channels
        if self.model.get_num_y_channels() != 1:
            return None

        if self.sigma is None:
            return None

        n = len(self.x) - len(self.free_parameters)

        if n < 1:
            raise ValueError(
                "Cannot calculate chi squared with "
                f"{len(self.free_parameters)} fit parameters and only "
                f"{len(self.x)} data points."
            )

        y_fit = self.evaluate()[1]
        chi_2 = np.sum(np.power((self.y - y_fit) / self.sigma, 2))
        p = stats.chi2.sf(chi_2, n)
        return p
