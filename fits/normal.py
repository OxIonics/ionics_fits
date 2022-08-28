import numpy as np
from scipy import optimize, stats
from . import FitBase


class NormalFit(FitBase):
    """Fit normally-distributed data.

    We use least-squares fitting as a maximum-likelihood parameter estimator for
    normally distributed data. For data that is close to normal this is usually a pretty
    good approximation of a true MLE estimator.
    """
    def _fit(self, x, y, initial_values, bounds, func, x_scale, y_scale):
        y_err = None if self._y_err is None else self._y_err / y_scale

        p0 = [initial_values[param] for param in self._free_params]
        lower = [bounds[param][0] for param in self._free_params]
        upper = [bounds[param][1] for param in self._free_params]

        assert x.dtype == np.float64
        assert y.dtype == np.float64

        p, p_cov = optimize.curve_fit(
            f=func,
            xdata=x,
            ydata=y,
            p0=p0,
            sigma=y_err,
            absolute_sigma=y_err is not None,
            bounds=(lower, upper),
            method="trf",
        )
        p_err = np.sqrt(np.diag(p_cov))

        p = {param: value for param, value in zip(self._free_params, p)}
        p_err = {param: value for param, value in zip(self._free_params, p_err)}

        return p, p_err

    def fit_significance(self) -> float:
        """Returns an estimate of the goodness of fit as a number between 0 and 1.

        This is the probability that the dataset could have arisen through chance under
        the assumption that the fitted model is correct and with the fit statistics. A
        value of `1` indicates a perfect fit (all data points lie on the fitted curve)
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

        y_fit = self.evaluate(self._x)[1]
        chi_2 = np.sum(np.power((self._y - y_fit) / self._y_err, 2))
        p = stats.chi2.sf(chi_2, n)
        return p

    def set_dataset(self, x, y, y_err=None):
        self._x = np.array(x, dtype=np.float64, copy=True)
        self._y = np.array(y, dtype=np.float64, copy=True)
        self._y_err = (
            None if y_err is None else np.array(y_err, dtype=np.float64, copy=True)
        )

        valid_pts = np.logical_and(np.isfinite(self._x), np.isfinite(self._y))
        self._x = self._x[valid_pts]
        self._y = self._y[valid_pts]
        self._y_err = None if self._y_err is None else self._y_err[valid_pts]

        inds = np.argsort(self._x)
        self._x = self._x[inds]
        self._y = self._y[inds]
        self._y_err = None if self._y_err is None else self._y_err[inds]

        self._fitted_params = None
        self._fitted_param_uncertainties = None

        if self._x.shape != self._y.shape:
            raise ValueError("Shapes of x and y do not match.")

        if self._y_err is not None and self._y.shape != self._y_err.shape:
            raise ValueError("Shape of y and y_err do not match")
