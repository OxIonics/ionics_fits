from numbers import Real

from numpy import isnan
from scipy.stats import beta

ONE_SIGMA = 0.682689492
ALPHA_2 = (1.0 - ONE_SIGMA) / 2.0


def binomial(k: Real | float, N: Real | float) -> tuple[float, float]:
    """Calculate the estimated source probability from measuring `k` successes
    out of `N` attempts, and its error (half width of the Clopper-Pearson
    interval)
    """
    if N <= 0:
        raise ValueError(f"N must be a positive number (N={N})")
    if k < 0 or k > N:
        raise ValueError(f"k must remain in the range [0; N] (k={k}, N={N})")

    kf = float(k)
    nf = float(N)

    # Formula from Wikipedia:
    # https://w.wiki/ETvb#Clopper%E2%80%93Pearson_interval
    lower = beta.ppf(ALPHA_2, kf, nf - kf + 1.0)
    upper = beta.ppf(1.0 - ALPHA_2, kf + 1.0, nf - kf)

    width = (1.0 if isnan(upper) else float(upper)) - (
        0.0 if isnan(lower) else float(lower)
    )

    return float(k) / float(N), width / 2.0
