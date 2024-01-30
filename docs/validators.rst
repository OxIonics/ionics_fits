.. _validators:

Fit Validators
=================

It's not enough to just fit the data, we want to know if we can trust the fit results
before acting on them.  There are two distinct aspects to the validation problem: did
the fit find the model parameters which best match the data (as opposed to getting stuck
in a local minimum in parameter space far from the global optimum)? and, are the fitted
parameter values consistent with our prior knowledge of the system (e.g. we know that a
fringe contrast must lie within certain bounds).

First, any prior knowledge about the system should be incorporated by specifying fixed
parameter values and parameter bounds. After that, the fit is validated using a
[``FitValidator``](../master/ionics_fits/validators.py). Validators provide a flexible
and extensible framework for using statistical tests to validate fits.

Todo: add a note about chi2 / n sigma validators.