from typing import Tuple
import numpy as np


import ionics_fits as fits


class FitValidator:
    """Base class for fit validators"""

    def window_fun(self, fit_results: fits.Fitter):
        """Returns a boolean array of x-axis points which should be included in
        fit validation. This implementation includes all x-axis data points;
        override to only include a subset.
        """
        return np.full(fit_results.x.shape, True)

    def validate(self, fit_results: fits.Fitter) -> Tuple[bool, float]:
        """Returns a Tuple of:

         bool: `True` if the fit was successful otherwise `False`.
         float: significance from the applied test as a number between 0.0 (
           complete failure) and 1.0 (perfect fit).

        Subclasses must override this method
        """
        raise NotImplementedError


class NSigmaValidator(FitValidator):
    def __init__(self, n_sigma: float = 3.0, significance_threshold: float = 0.75):
        """Fit validator which checks that at least
        :param significance_threshold: of points lie within :param n_sigma:
        standard errors of the fitted value.

        This is a relatively forgiving (easy to configure in a way that gives
        minimal "good" fits which fail validation) general-purpose fit validator.
        """
        self.n_sigma = n_sigma
        self.significance_threshold = significance_threshold

    def validate(self, fit_results: fits.Fitter) -> Tuple[bool, float]:
        if fit_results.sigma is None:
            raise ValueError(
                "Cannot validate fit without standard errors for each point"
            )

        window = self.window_fun(fit_results)

        x = fit_results.x[window]
        y = fit_results.y[:, window]
        sigma = fit_results.sigma[:, window]

        _, y_fit = fit_results.evaluate(x_fit=x)

        errs = np.abs((y - y_fit) / sigma)
        errs = np.sort(errs.flatten())

        good_points = errs < self.n_sigma
        good_fraction = sum(good_points) / len(good_points)

        return good_fraction >= self.significance_threshold, good_fraction
