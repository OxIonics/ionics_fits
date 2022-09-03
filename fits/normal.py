import logging
import numpy as np
from scipy import optimize, stats
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from . import FitBase
from .utils import Array, ArrayLike

if TYPE_CHECKING:
    num_samples = float
    num_values = float


logger = logging.getLogger(__name__)


class NormalFit(FitBase):
    """Fit normally-distributed data.

    We use least-squares fitting as a maximum-likelihood parameter estimator for
    normally distributed data. For data that is close to normal this is usually a pretty
    good approximation of a true MLE estimator. YMMV...
    """

    def _fit(
        self,
        x: Array[("num_samples",), np.float64],
        y: Array[("num_samples",), np.float64],
        initial_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        free_func: Callable[
            # TODO: correct annotation for *args?
            [Array[("num_samples",), np.float64], List[float]],
            Array[("num_samples",), np.float64],
        ],
        x_scale: Optional[float],
        y_scale: Optional[float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Least-squares data fitting of normally-distributed data.

        :param x: x-axis data
        :param y: y-axis data
        :param initial_values: dictionary mapping model parameter names to initial
            values (either user-specified or from heuristics) to use as a starting point
            for the optimizer.
        :param bounds: dictionary mapping model parameter names to their
            `(lower, upper)` bounds. Fitted values must lie within these bounds.
        :param free_func: wrapper for the model function, taking only values for the
            fit's free parameters.
        :param x_scale: x-axis scale factor or `None` if the axis was not rescaled
        :param y_scale: y-axis scale factor or `None` if the axis was not rescaled

        :returns: tuple of dictionaries mapping model parameter names to their fitted
            values and uncertainties.
        """
        y_err = self._y_err
        if y_scale is not None:
            y_err = None if y_err is None else y_err / y_scale

        p0 = [initial_values[param] for param in self._free_params]
        lower = [bounds[param][0] for param in self._free_params]
        upper = [bounds[param][1] for param in self._free_params]

        assert x.dtype == np.float64
        assert y.dtype == np.float64

        p, p_cov, infodict, mesg, ier = optimize.curve_fit(
            f=free_func,
            xdata=x,
            ydata=y,
            p0=p0,
            sigma=y_err,
            absolute_sigma=y_err is not None,
            bounds=(lower, upper),
            method="trf",
            full_output=True,
        )

        p_err = np.sqrt(np.diag(p_cov))

        p = {param: value for param, value in zip(self._free_params, p)}
        p_err = {param: value for param, value in zip(self._free_params, p_err)}

        logger.debug(
            f"Least-squares fit complete: " f"{infodict}\n" f"{mesg}\n" f"{ier}"
        )

        return p, p_err

    def fit_significance(self) -> float:
        """Returns an estimate of the goodness of fit as a number between 0 and 1.

        Implemented using the Chi-Squared.

        This is the defined as the probability that fit residuals as large as the ones
        we observe could have arisen through chance given our assumed statistics and
        assuming that the fitted model perfectly represents the probability distribution

        A value of `1` indicates a perfect fit (all data points lie on the fitted curve)
        a value close to 0 indicates significant deviations of the dataset from the
        fitted model.
        """
        if self._y_err is None:
            raise ValueError("Cannot calculate fit significance without knowing y_err")

        n = len(self._x) - len(self._free_params)

        if n < 1:
            raise ValueError(
                "Cannot calculate chi squared with "
                f"{len(self._free_params)} fit parameters and only "
                f"{len(self._x)} data points."
            )

        y_fit = self.evaluate()[1]
        chi_2 = np.sum(np.power((self._y - y_fit) / self._y_err, 2))
        p = stats.chi2.sf(chi_2, n)
        return p

    def set_dataset(
        self,
        x: ArrayLike[("num_samples",), np.float64],
        y: ArrayLike[("num_samples",), np.float64],
        y_err: Optional[ArrayLike[("num_samples",), np.float64]] = None,
    ):
        """Sets the dataset to be fit.

        :param x: x-axis data
        :param y: y-axis data
        :param y_err: optional y-axis standard deviations
        """
        self._x = np.array(x, dtype=np.float64, copy=True)
        self._y = np.array(y, dtype=np.float64, copy=True)
        self._y_err = (
            None if y_err is None else np.array(y_err, dtype=np.float64, copy=True)
        )

        valid_pts = np.logical_and(np.isfinite(self._x), np.isfinite(self._y))
        if self._y_err is not None:
            valid_pts = np.logical_and(valid_pts, np.isfinite(self.y_err))

        self._x = self._x[valid_pts]
        self._y = self._y[valid_pts]
        self._y_err = None if self._y_err is None else self._y_err[valid_pts]

        if self._x.shape != self._y.shape:
            raise ValueError("Shapes of x and y must match.")

        if self._y_err is not None and self._y_err.shape != self._y.shape:
            raise ValueError("Shapes of y_err and y must match.")

        inds = np.argsort(self._x)
        self._x = self._x[inds]
        self._y = self._y[inds]
        self._y_err = None if self._y_err is None else self._y_err[inds]

        self._estimated_values = None
        self._fitted_params = None
        self._fitted_param_uncertainties = None