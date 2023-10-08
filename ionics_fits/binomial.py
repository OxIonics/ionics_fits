import copy
import logging
import numpy as np
import pprint
from scipy import optimize, stats
from typing import Callable, Dict, Tuple, TYPE_CHECKING

from . import Fitter, Model, ModelParameter
from .utils import Array, ArrayLike

if TYPE_CHECKING:
    num_samples = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


logger = logging.getLogger(__name__)


class BinomialFitter(Fitter):
    """Fitter for Binomially-distributed data.

    The model is interpreted as giving the success probability for a Bernoulli
    trial under a given set of parameters: `p = M(x; params)`

    The y-axis data is interpreted as the success fraction, such that the total
    number of successes is equal to `k = y * num_trails`.
    """

    def __init__(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_y_channels", "num_samples"), np.float64],
        num_trials: int,
        model: Model,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults.
        :param num_trials: number of Bernoulli trails for each sample
        """
        self.num_trials = num_trials

        # https://github.com/OxIonics/ionics_fits/issues/105
        model = copy.deepcopy(model)
        for parameter in model.parameters.values():
            parameter.scale_func = lambda x_scale, y_scale, _: None

        super().__init__(x=x, y=y, model=model)

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
            "Starting binomial fitting with initial parameters: "
            f"{pprint.pformat(p0_dict)}"
        )

        def cost_fun(free_param_values: float) -> float:
            p = free_func(x, *free_param_values.tolist())

            if any(p < 0) or any(p > 1):
                raise RuntimeError("Model values must lie between 0 and 1")

            if np.any(y < 0) or np.any(y > 1):
                raise RuntimeError("y values must lie between 0 and 1")

            n = self.num_trials
            k = np.rint(y * n, out=np.zeros_like(y, dtype=int), casting="unsafe")
            logP = stats.binom.logpmf(k=k, n=n, p=p)
            C = -np.sum(logP)
            return C

        res = optimize.minimize(
            fun=cost_fun,
            x0=p0,
            bounds=zip(lower, upper),
        )

        if not res.success:
            raise RuntimeError(f"MLE fit failed: {res.message}")

        p = {param: value for param, value in zip(free_parameters, res.x)}
        p_err = {param: np.nan for param in free_parameters}

        # p_err = np.sqrt(np.diag(p_cov))
        # p_err = {param: value for param, value in zip(free_parameters, p_err)}

        return p, p_err
