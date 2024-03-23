import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from statsmodels.stats import proportion


from .common import Model, TJACOBIAN, TX, TY
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

    def cost_func(
        self,
        free_param_values: Array[("num_free_params",), np.float64],
        x: TX,
        y: TY,
        free_func: Callable[..., TY],
        jacobian_func: Callable[[List[int]], TJACOBIAN],
    ) -> (float, Array[("num_free_params",), np.float64]):
        p = free_func(x, *free_param_values.tolist())

        # avoid divide by zero issues when p is exactly 0 or 1
        eps = 1e-12
        p[p == 0] = eps
        p[p == 1] = 1 - eps

        if np.any(p < 0) or np.any(p > 1):
            raise RuntimeError("Model values must lie between 0 and 1")

        n = self.num_trials
        k = y * n

        # The negative Log-Likelihood for a Binomial distribution is:
        # L = -sum(log(BinomPMF(k, n, p)))
        # L = -sum(log(Choose(n, k)) + k*log(p) + (n-k)*log(q))
        # L = -sum(log(Choose(n, k))) - sum(k*log(p) + (n-k)*log(q)))
        #
        # where: BinomPMF = Choose(n, k) * p^k * q^(n - k)
        #
        # Since the combinatorial part is just a constant offset we do not need to
        # include it in our cost function. We can thus simplify to:
        #
        # C = -sum(k*log(p) + (n-k)*log(q)
        C = -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

        # Calculate Jacobian:
        #
        # C = -sum(k*log(p) + (n-k)*log(1 - p)
        # dC/d(param) = -sum(k / p * dp/d(param) - (n-k) / (1 - p) * dp/d(param))
        # dC/d(param) = sum(dp/d(param) * ((n-k) / (1 - p) - k / p))

        model_jac = jacobian_func(free_param_values)
        jac = np.sum(model_jac * ((n - k) / (1 - p) - k / p), axis=(1, 2))

        return (C, jac)

    def hessian(
        self, x: TX, y: TY, param_values: Dict[str, float], free_params: List[int]
    ) -> Array[("num_free_params", "num_free_params"), np.float64]:
        # dC/d(param) = sum(dp/d(param) * ((n-k) / (1 - p) - k / p))
        #
        # d2C/d(param_i)d(param_j) = sum(
        #  d2p/d(param_i)d(param_j) * ((n-k) / (1 - p) - k / p)
        #  + dp/d(param_i) * dp/d(param_j) * (k/p^2 + (n-k)/(1-p)^2)
        # )
        #
        # d2C/d(param_i)d(param_j) = sum(
        #  A * d2p/d(param_i)d(param_j)
        #  + B * dp/d(param_i) * dp/d(param_j)
        # )
        #
        # where:
        #  A = ((n-k) / (1 - p) - k / p)
        #  B = (k/p^2 + (n-k) / (1-p)^2)

        x = np.atleast_2d(x)
        p = self.model.func(x, param_values)
        n = self.num_trials
        k = y * n

        # avoid divide by zero issues when p is exactly 0 or 1
        eps = 1e-12
        p[p == 0] = eps
        p[p == 1] = 1 - eps

        A = (n - k) / (1 - p) - k / p
        B = k / (p**2) + (n - k) / ((1 - p) ** 2)

        model_jacobian = self.model.jacobian(
            x=x, param_values=param_values, included_params=free_params
        )
        model_hessian = self.model.hessian(
            x=x, param_values=param_values, included_params=free_params
        )

        num_free_params = len(free_params)
        hessian = np.zeros((num_free_params, num_free_params) + x.shape)

        for i_idx in range(num_free_params):
            for j_idx in range(num_free_params):
                hessian[i_idx, j_idx, :, :] = (
                    A * model_hessian[i_idx, j_idx, :, :]
                    + B * model_jacobian[i_idx, :, :] * model_jacobian[j_idx, :, :]
                )

        hessian = np.sum(hessian, axis=(2, 3))

        return hessian

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
