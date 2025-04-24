r"""It's not enough to just fit data, we want to know whether we can trust the fit
results. This is where validators come in.

There are two distinct aspects to the validation problem:

* did the fit find the model parameters which best match the data (as opposed to getting
  stuck in a local minimum in parameter space far from the global optimum)?
* Are the fitted parameter values consistent with our prior knowledge of the system
  (e.g. we know that a fringe contrast must lie within certain bounds).

The approach taken in ``ionics_fits`` is as follows. First any prior knowledge about the
system should be encoded into the :class:`~ionics_fits.common.ModelParameter`
configuration by setting parameter bounds, fixing parameters, etc.

After that, the fit is validated using a :class:`FitValidator`. Validators provide a
flexible and extensible framework for using statistical tests to validate fits.
"""

from typing import Tuple

import numpy as np

from .common import Fitter


class FitValidator:
    """Base class for fit validators"""

    def window_fun(self, fit_results: Fitter):
        """Returns a boolean array of x-axis points which should be included in
        fit validation. This implementation includes all x-axis data points;
        override to only include a subset.
        """
        return np.full(fit_results.x.shape[1], True)

    def validate(self, fit_results: Fitter) -> Tuple[bool, float]:
        """Validates the fit.

        Subclasses must override this method

        :returns: tuple specifying whether the fit succeeded and the fit significance
          as a number between 0 (complete failure) and 1 (perfect fit).
        """
        raise NotImplementedError


class NSigmaValidator(FitValidator):
    def __init__(self, n_sigma: float = 3.0, significance_threshold: float = 0.75):
        """Fit validator which checks that at least
        ``significance_threshold`` of points lie within ``n_sigma``
        standard errors of the fitted value.

        This is a relatively forgiving (easy to configure in a way that gives
        minimal "good" fits which fail validation) general-purpose fit validator.

        :param n_sigma: number of standard errors that points allowed to differ from the
          model.
        :param significance_threshold: fraction of points which must lie within
          ``n_sigma`` of the model for the fit to be considered successful.
        """
        self.n_sigma = n_sigma
        self.significance_threshold = significance_threshold

    def validate(self, fit_results: Fitter) -> Tuple[bool, float]:
        if fit_results.sigma is None:
            raise ValueError(
                "Cannot validate fit without standard errors for each point"
            )

        window = self.window_fun(fit_results)

        x = fit_results.x[:, window]
        y = fit_results.y[:, window]
        sigma = fit_results.sigma[:, window]

        _, y_fit = fit_results.evaluate(x_fit=x)

        errs = np.abs((y - y_fit) / sigma)
        errs = np.sort(errs.ravel())

        good_points = errs < self.n_sigma
        good_fraction = sum(good_points) / len(good_points)

        return good_fraction >= self.significance_threshold, good_fraction
