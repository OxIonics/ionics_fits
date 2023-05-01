import numpy as np
from scipy import special
from typing import TYPE_CHECKING

from ..common import Array, ModelParameter


if TYPE_CHECKING:
    num_fock_states = float


# pytype: disable=invalid-annotation
def thermal_state_probs(
    n_max: int, n_bar: ModelParameter(lower_bound=0)
) -> Array[("num_fock_states",), np.float64]:
    """Returns an array of Fock state occupation probabilities for a thermal state of
    mean occupancy :param n_bar:, truncated at a maximum Fock state of |n_max>
    """
    n = np.arange(n_max + 1, dtype=int)
    return np.power(n_bar / (n_bar + 1), n) / (n_bar + 1)


# pytype: enable=invalid-annotation

# pytype: disable=invalid-annotation
def coherent_state_probs(
    n_max: int, alpha: ModelParameter(lower_bound=0)
) -> Array[("num_fock_states",), np.float64]:
    """Returns an array of the Fock state occupation probabilities for a coherent
    state described by :param alpha:, truncated at a maximum Fock state of |n_max>
    """
    n = np.arange(n_max + 1, dtype=int)
    n_bar = np.power(np.abs(alpha), 2)

    # The standard expression is: P_n = n_bar**n * exp(-n_bar) / n!
    # However, this is difficult to evaluate as n increases, so we use an alternate form
    # NB for integer n: gamma(n + 1) = n!
    if n_bar == 0:
        P_n = np.zeros_like(n)
        P_n[0] = 1
    else:
        P_n = np.exp(n * np.log(n_bar) - n_bar - special.gammaln(n + 1))
    return P_n


# pytype: enable=invalid-annotation
