import logging
import pprint
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import optimize, stats

from .common import TX, TY, Fitter, Model, ModelParameter
from .utils import Array

if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_samples_flattened = float
    num_x_axes = float

logger = logging.getLogger(__name__)


class NormalFitter(Fitter):
    """Fitter for Normally-distributed data.

    We use least-squares fitting as a maximum-likelihood parameter estimator for
    normally distributed data. For data that is close to normal this is usually
    a pretty good approximation of a true MLE estimator.

    See :class:`~ionics_fits.common.Fitter` for further details.
    """

    def __init__(
        self,
        x: TX,
        y: TY,
        model: Model,
        sigma: Optional[TY] = None,
        curve_fit_args: Optional[Dict] = None,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param sigma: optional y-axis standard deviations.
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults. The
            model is (deep) copied and stored as an attribute.
        :param curve_fit_args: optional dictionary of keyword arguments to be passed
            into ``scipy.curve_fit``.
        """
        if sigma is None:
            self.sigma = None
        else:
            self.sigma = np.array(sigma, dtype=np.float64, copy=True)

        self.curve_fit_args = {"method": "trf"}
        if curve_fit_args is not None:
            self.curve_fit_args.update(curve_fit_args)

        super().__init__(x=x, y=y, model=model)

    def _fit(
        self,
        x: TX,
        y: TY,
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., TY],
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Array[("num_free_params", "num_free_params"), np.float64],
    ]:
        sigma = None if self.sigma is None else self.sigma / self.y_scales[:, None]

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

        logger.debug(
            "Starting least-squares fitting with initial parameters: "
            f"{pprint.pformat(p0_dict)}"
        )

        def fit_func(
            _: Array[
                (
                    "num_x_axes",
                    "num_samples_flattened",
                ),
                np.float64,
            ],
            *free_param_values: float,
        ) -> Array[("num_samples_flattened",), np.float64]:
            """Call the model function with the values of the free parameters."""
            return np.ravel(free_func(x, *free_param_values))

        y_flat = np.ravel(y)
        sigma_flat = None if sigma is None else np.ravel(sigma)

        p, p_cov = optimize.curve_fit(
            f=fit_func,
            xdata=y_flat,  # not used during the fit
            ydata=y_flat,
            p0=p0,
            sigma=sigma_flat,
            absolute_sigma=sigma is not None,
            bounds=(lower, upper),
            **self.curve_fit_args,
        )

        p_err = np.sqrt(np.diag(p_cov))

        p = {param: value for param, value in zip(free_parameters, p)}
        p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err, p_cov

    def calc_sigma(self) -> Optional[TX]:
        """Returns an array of standard error values for each y-axis data point
        if available.
        """
        return self.sigma

    def chi_squared(self, x: TX, y: TY, sigma: TY) -> float:
        """Returns the Chi-squared fit significance for the fitted model compared to a
        given dataset as a number between 0 and 1.

        The significance gives the probability that fit residuals as large as the ones
        we observe could have arisen through chance given our assumed statistics and
        assuming that the fitted model perfectly represents the probability
        distribution.

        A value of ``1`` indicates a perfect fit (all data points lie on the fitted
        curve) a value close to ``0`` indicates super-statistical deviations of the
        dataset from the fitted model.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        sigma = np.atleast_2d(sigma)

        if sigma.shape != y.shape:
            raise ValueError(
                f"Mismatch between shapes of sigma ({sigma.shape}) and y ({y.shape})"
            )

        n = y.size - len(self.free_parameters)

        if n < 1:
            raise ValueError(
                "Cannot calculate chi squared for fit with "
                f"{len(self.free_parameters)} floated parameters and only "
                f"{y.size} data points."
            )

        y_fit = self.model.func(x, self.values)
        chi_2 = np.sum(((y - y_fit) / sigma) ** 2)
        p = stats.chi2.sf(chi_2, n)

        return p
