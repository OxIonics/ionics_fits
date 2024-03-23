import copy
import logging
import numpy as np
import pprint
from scipy import optimize
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .common import Fitter, Model, ModelParameter, TJACOBIAN, TX, TY
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

    def cost_func(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: TX,
        y: TY,
        free_func: Callable[..., TY],
        jacobian_func: Callable[[List[int]], TJACOBIAN],
    ) -> (float, Array[("num_free_params",), np.float64]):
        """Cost function used during fitting.

        The cost function is based on the negative log-likelihood of the dataset given
        a set of values for the model parameters. It can deviate from the exact
        log-likelihood function (for example to make it faster to calculate /
        numerically more stable) so long as whenever the cost is minimized the
        log-likelihood is maximised.

        This function must be overridden by specialisations of
            :class:`~ionics_fits.MLE.MLEFitter`.

        :param free_param_values: array of floated parameter values
        :param x: x-axis data
        :param y: y-axis data
        :param free_func: convenience wrapper for the model function, taking only values
            for the fit's free parameters
        :param jacobian_func: convenience wrapper for the model's Jacobian function,
            including only the fit's free parameters
        :returns: tuple giving the values of the cost function and its Jacobian
        """
        raise NotImplementedError

    def hessian(
        self, x: TX, y: TY, param_values: Dict[str, float], free_params: List[int]
    ) -> Array[("num_free_params", "num_free_params"), np.float64]:
        """Hessian of the cost function, used to calculate the parameter covariance
        matrix.

        This function must be overridden by specialisations of
            :class:`~ionics_fits.MLE.MLEFitter`.

        :param x: x-axis data
        :param y: y-axis data
        :param param_values: dictionary of fitted model parameter values
        :param free_params: list of free parameters for the fit
        :returns: the Hessian matrix
        """
        raise NotImplementedError

    def _fit(
        self,
        x: TX,
        y: TY,
        parameters: Dict[str, ModelParameter],
        free_func: Callable[..., TY],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
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
        fixed_params = {
            param_name: param_data.fixed_to
            for param_name, param_data in parameters.items()
            if param_data.fixed_to is not None
        }
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

        def model_jac_func(free_param_values):
            param_values = dict(fixed_params)
            param_values.update(
                {
                    free_parameters[idx]: free_param_values[idx]
                    for idx in range(num_free_params)
                }
            )
            model_jac = self.model.jacobian(
                x=x,
                param_values=param_values,
                included_params=free_parameters,
            )
            return model_jac

        # maxls setting helps with ABNORMAL_TERMINATION_IN_LNSRCH
        res = optimize.minimize(
            fun=self.cost_func,
            jac=True,
            args=(x, y, free_func, model_jac_func),
            x0=p0,
            bounds=zip(lower, upper),
            **self.minimizer_args,
        )

        if not res.success:
            raise RuntimeError(f"{self.TYPE} fit failed: {res.message}")

        p = {param: value for param, value in zip(free_parameters, res.x)}

        # Compute parameter covariance matrix from the inverse of the Hessian at the
        # optimum.
        param_values = dict(p)
        param_values.update(fixed_params)

        hessian = self.hessian(
            x=x, y=y, param_values=param_values, free_params=free_parameters
        )
        p_cov = np.linalg.inv(hessian)
        p_err = np.sqrt(np.diag(p_cov))

        p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err
