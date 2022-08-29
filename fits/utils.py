import numpy as np
from statsmodels.stats.proportion import proportion_confint


# From https://github.com/oxfordiontrapgroup/oitg
def uncertainty_to_string(x, err, precision=1):
    """Returns a string representing nominal value x with uncertainty err.
    Precision is the number of significant digits in uncertainty

    Returns the shortest string representation of `x +- err` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)

    Based on http://stackoverflow.com/questions/6671053/python-pretty-print-errorbars"""

    if np.isnan(x) or np.isnan(err):
        return "NaN"

    if np.isinf(x) or np.isinf(err):
        return "inf"

    # Chuck away sign of err
    err = abs(err)

    # An error of 0 is not meaningful
    assert err > 0

    # base 10 exponents
    err_exp = int(np.floor(np.log10(err)))

    # If x is 0 set the x_exp to be the same as err_exp, meaning that the
    # result is formatted as 0(err)
    try:
        x_exp = int(np.floor(np.log10(abs(x))))
    except ValueError:
        x_exp = err_exp

    # Or if |x| < err, do the same
    if abs(x) < err:
        x_exp = err_exp

    # uncertainty
    un_exp = err_exp - precision + 1
    un_int = round(err * 10 ** (-un_exp))

    # nominal value
    no_exp = un_exp
    no_int = round(x * 10 ** (-no_exp))

    # format - nom(unc)exp
    fieldw = x_exp - no_exp
    fmt = "%%.%df" % fieldw

    result1 = (fmt + "(%.0f)e%d") % (no_int * 10 ** (-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = "%%.%df" % fieldw
    result2 = (fmt + "(%.0f)") % (no_int * 10**no_exp, un_int * 10 ** max(0, un_exp))

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1


# From https://github.com/oxfordiontrapgroup/oitg
def binom_twosided(k, N):
    """Returns the estimated source probability and confidence interval from
    a sample of k Trues out of N samples.
    """
    if k > N or k < 0:
        raise ValueError("k must be between 0 and N (k={}, N={})".format(k, N))

    # 'beta' is Clopper–Pearson method; chosen alpha corresponds to 1σ of a normal
    # distribution (68%).
    confint = proportion_confint(k, N, alpha=0.3173, method="beta")

    # Strip out NaNs for confidence intervals at the boundary
    if np.isnan(confint[0]):
        confint = (0, confint[1])
    elif np.isnan(confint[1]):
        confint = (confint[0], 1)

    p = k / N
    return p, confint


# From https://github.com/oxfordiontrapgroup/oitg
def binom_onesided(k, N):
    """Returns the estimated source probability and uncertainty from a sample
    of k Trues out of N samples.
    """
    p, confint = binom_twosided(k, N)
    # This is not great at the boundary (k=0 or k=N).
    # TODO: numerically test this, perhaps add a correction term
    # (Laplace law of succession?)
    uncertainty = (confint[1] - confint[0]) / 2

    return p, uncertainty
