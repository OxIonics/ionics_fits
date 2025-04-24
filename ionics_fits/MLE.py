import copy
import logging
import pprint
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from .common import TX, TY, Fitter, Model, ModelParameter
from .utils import Array

if TYPE_CHECKING:
    num_free_params = float


logger = logging.getLogger(__name__)


class MLEFitter(Fitter):
    """Base class for maximum Likelihood Parameter Estimation fitters.

    Implementations should override the :meth:`log_likelihood` and
    :meth:`~ionics_fits.common.Fitter.calc_sigma` methods.

    See :class:`~ionics_fits.common.Fitter` for further details.
    """

    TYPE: str = "MLE"

    def __init__(
        self,
        x: TX,
        y: TY,
        model: Model,
        step_size: float = 1e-4,
        minimizer_args: Optional[Dict] = None,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults. The
            model is (deep) copied and stored as an attribute.
        :param step_size: step size used when calculating the log likelihood's Hessian
            as part of finding the fitted parameter standard errors. Where finite
            parameter bounds are provided, they are used to scale the step size
            appropriately for each parameter.
        :param minimizer_args: optional dictionary of keyword arguments to be passed
            into ``scipy.optimize.minimize``. By default we set ``maxls`` to 100.
        """
        if minimizer_args is not None:
            self.minimizer_args = dict(minimizer_args)
        else:
            self.minimizer_args = {"options": {"maxls": 100}}

        self.step_size = step_size

        if np.any(y < 0) or np.any(y > 1):
            raise RuntimeError("y values must lie between 0 and 1")

        # Since we interpret the y-axis as a probability distribution, it should not be
        # rescaled
        def can_rescale() -> Tuple[List[bool], List[bool]]:
            rescale_xs, rescale_ys = self._can_rescale()
            return rescale_xs, [False] * len(rescale_ys)

        model = copy.deepcopy(model)
        self._can_rescale = model.can_rescale
        model.can_rescale = can_rescale

        super().__init__(x=x, y=y, model=model)

    def log_likelihood(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: TX,
        y: TY,
        free_func: Callable[..., TY],
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
        x: TX,
        y: TY,
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., TY],
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Array[("num_free_params", "num_free_params"), np.float64],
    ]:
        """Performs maximum likelihood parameter estimation and calculates standard
        errors in each parameter.

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
        num_free_params = len(free_parameters)

        p0 = [parameters[param].get_initial_value() for param in free_parameters]
        lower = [parameters[param].lower_bound for param in free_parameters]
        upper = [parameters[param].upper_bound for param in free_parameters]
        p0_dict = {
            param: parameters[param].get_initial_value() for param in free_parameters
        }

        logger.debug(
            f"Starting {self.TYPE} fitting with initial parameters: "
            f"{pprint.pformat(p0_dict)}"
        )

        # maxls setting helps with ABNORMAL_TERMINATION_IN_LNSRCH
        res = optimize.minimize(
            fun=self.log_likelihood,
            args=(x, y, free_func),
            x0=p0,
            bounds=zip(lower, upper),
            **self.minimizer_args,
        )

        if not res.success:
            raise RuntimeError(f"{self.TYPE} fit failed: {res.message}")

        # Compute parameter covariance matrix
        #
        # The covariance matrix is given by the inverse of the Hessian at the optimum.
        #
        # While scipy.minimize provides an approximate value for the inverse
        # Hessian, it's not accurate enough to be used for error estimation.
        # we perform our own calculation based on finite differences using the
        # specified step size, rescaled by the parameter bounds where possible.
        lower = [bound if bound is not None else -np.inf for bound in lower]
        upper = [bound if bound is not None else +np.inf for bound in upper]

        def diff(param_idx, fun):
            if np.isfinite(lower[param_idx]) and np.isfinite(upper[param_idx]):
                param_range = upper[param_idx] - lower[param_idx]
                step_size = self.step_size * param_range
            else:
                step_size = self.step_size

            def _fun(free_param_values):
                param_value = free_param_values[param_idx]
                param_upper = np.clip(
                    param_value + 0.5 * step_size,
                    a_min=lower[param_idx],
                    a_max=upper[param_idx],
                )
                param_lower = np.clip(
                    param_value - 0.5 * step_size,
                    a_min=lower[param_idx],
                    a_max=upper[param_idx],
                )
                delta = param_upper - param_lower

                param_upper_values = np.copy(free_param_values)
                param_lower_values = np.copy(free_param_values)
                param_upper_values[param_idx] = param_upper
                param_lower_values[param_idx] = param_lower

                f_upper = fun(param_upper_values)
                f_lower = fun(param_lower_values)

                return (f_upper - f_lower) / delta

            return _fun

        def log_likelihood(free_param_values):
            return self.log_likelihood(
                free_param_values=free_param_values, x=x, y=y, free_func=free_func
            )

        hessian = np.zeros((num_free_params, num_free_params))

        for i_idx in range(num_free_params):
            for j_idx in range(num_free_params):
                first_diff = diff(i_idx, log_likelihood)
                second_diff = diff(j_idx, first_diff)
                hessian[i_idx, j_idx] = second_diff(res.x)

        p_cov = np.linalg.inv(hessian)
        p_err = np.sqrt(np.diag(p_cov))

        p = {param: value for param, value in zip(free_parameters, res.x)}
        p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err, p_cov
