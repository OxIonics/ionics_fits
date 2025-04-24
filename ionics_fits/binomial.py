import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional

import numpy as np
from scipy import stats
from statsmodels.stats import proportion

from .common import TX, TY, Model
from .MLE import MLEFitter
from .utils import Array

if TYPE_CHECKING:
    num_free_params = float

logger = logging.getLogger(__name__)


class BinomialFitter(MLEFitter):
    """Maximum-likelihood parameter estimator for Binomially-distributed data.

    The model is interpreted as giving the success probability for a Bernoulli
    trial under a given set of parameters: ``p = M(x; params)``.

    The y-axis data is interpreted as the success fraction, such that the total
    number of successes is equal to ``k = y * num_trails``.

    See :class:`~ionics_fits.common.Fitter` and :class:`~ionics_fits.MLE.MLEFitter` for
    further details.
    """

    TYPE: str = "Binomial"

    def __init__(
        self,
        x: TX,
        y: TY,
        num_trials: int,
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
        :param num_trials: number of Bernoulli trails for each sample
        :param step_size: see :class:`~ionics_fits.MLE.MLEFitter`.
        :param minimizer_args: optional dictionary of keyword arguments to be passed
            into ``scipy.optimize.minimize``.
        """
        self.num_trials = num_trials
        super().__init__(
            x=x, y=y, model=model, step_size=step_size, minimizer_args=minimizer_args
        )

    def log_likelihood(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: TX,
        y: TY,
        free_func: Callable[..., TY],
    ) -> float:
        p = free_func(x, *free_param_values.tolist())

        if np.any(p < 0) or np.any(p > 1):
            raise RuntimeError("Model values must lie between 0 and 1")

        n = self.num_trials
        k = np.rint(y * n, out=np.zeros_like(y, dtype=int), casting="unsafe")
        logP = stats.binom.logpmf(k=k, n=n, p=p)
        C = -np.sum(logP)

        return C

    def calc_sigma(self) -> TY:
        """Return an array of standard error values for each y-axis data point."""
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

        sigma = 0.5 * (upper - lower)
        return sigma
