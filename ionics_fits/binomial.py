import logging
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
from scipy import stats
from statsmodels.stats import proportion


from . import MLEFitter, Model
from .utils import Array, ArrayLike

if TYPE_CHECKING:
    num_free_params = float
    num_samples = float
    num_values = float
    num_y_channels = float
    num_samples_flattened = float


logger = logging.getLogger(__name__)


class BinomialFitter(MLEFitter):
    """Maximum-likelihood parameter estimator for Binomially-distributed data.

    The model is interpreted as giving the success probability for a Bernoulli
    trial under a given set of parameters: `p = M(x; params)`

    The y-axis data is interpreted as the success fraction, such that the total
    number of successes is equal to `k = y * num_trails`.
    """

    TYPE: str = "Binomial"

    def __init__(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_y_channels", "num_samples"), np.float64],
        num_trials: int,
        model: Model,
        step_size: float = 1e-4,
    ):
        """Fits a model to a dataset and stores the results.

        :param x: x-axis data
        :param y: y-axis data
        :param model: the model function to fit to. The model's parameter dictionary is
            used to configure the fit (set parameter bounds etc). Modify this before
            fitting to change the fit behaviour from the model class' defaults.
        :param num_trials: number of Bernoulli trails for each sample
        :param step_size: see :class MLEFitter:
        """
        self.num_trials = num_trials
        super().__init__(x=x, y=y, model=model, step_size=step_size)

    def log_liklihood(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_y_channels", "num_samples"), np.float64],
        free_func: Callable[..., Array[("num_y_channels", "num_samples"), np.float64]],
    ) -> float:
        p = free_func(x, *free_param_values.tolist())

        if any(p < 0) or any(p > 1):
            raise RuntimeError("Model values must lie between 0 and 1")

        n = self.num_trials
        k = np.rint(y * n, out=np.zeros_like(y, dtype=int), casting="unsafe")
        logP = stats.binom.logpmf(k=k, n=n, p=p)
        C = -np.sum(logP)

        return C

    def calc_sigma(
        self,
    ) -> Optional[Array[("num_y_channels", "num_samples"), np.float64]]:
        """Return an array of standard error values for each y-axis data point
        if available.
        """
        k = np.rint(
            self.y * self.num_trials,
            out=np.zeros_like(self.y, dtype=int),
            casting="unsafe",
        )

        lower, upper = proportion.proportion_confint(
            count=k,
            nobs=self.num_trials,
            alpha=1 - 0.6827,  # 1 sigma for Normal distributions
            method="beta",
        )

        # Replace NaNs for points where k={0, num_trials}
        lower[np.isnan(lower)] = 0
        upper[np.isnan(upper)] = 1

        sigma = 0.5 * (lower + upper)
        return sigma
