import numpy as np
from scipy import special
from typing import TYPE_CHECKING

from ..common import Array, ModelParameter
from ..utils import scale_invariant

if TYPE_CHECKING:
    num_fock_states = float


# pytype: disable=invalid-annotation
def thermal_state_probs(
    n_max: int, n_bar: ModelParameter(lower_bound=0, scale_func=scale_invariant)
) -> Array[("num_fock_states",), np.float64]:
    """Thermal state probability distribution.

    :param n_max: the distribution is truncated at a maximum Fock state of ``|n_max>``
    :param n_bar: the mean Fock state occupation
    :returns: array of Fock state occupation probabilities
    """
    n = np.arange(n_max + 1, dtype=int)
    return ((n_bar / (n_bar + 1)) ** n) / (n_bar + 1)


# pytype: enable=invalid-annotation

# pytype: disable=invalid-annotation
def coherent_state_probs(
    n_max: int, alpha: ModelParameter(lower_bound=0, scale_func=scale_invariant)
) -> Array[("num_fock_states",), np.float64]:
    """Coherent state probability distribution.

    A coherent state is defined as::

        |α> = exp(α a_dag - α* a) |0>

    where ``a_dag`` and ``a`` denote the harmonic-oscillator creation and annihilation
    operators. The mean Fock state occupancy is given by ``n_bar = |α|^2``

    :param n_max: the distribution is truncated at a maximum Fock state of ``|n_max>``
    :param alpha: Complex displacement parameter
    :returns: array of Fock state occupation probabilities
    """
    n = np.arange(n_max + 1, dtype=int)
    n_bar = np.abs(alpha) ** 2

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


# pytype: disable=invalid-annotation
def squeezed_state_probs(
    n_max: int, zeta: ModelParameter(lower_bound=0, scale_func=scale_invariant)
) -> Array[("num_fock_states",), np.float64]:
    """Squeezed state probability distribution.

    A pure squeezed state is defined as::

        |ζ> = exp[1 / 2 *  (ζ* a^2 - ζ a_dag^2)] |0>,

    where ``a_dag`` and ``a`` denote the harmonic-oscillator creation and annihilation
    operators.

    :param n_max: the distribution is truncated at a maximum Fock state of ``|n_max>``
    :param zeta: Complex squeezing parameter
    :returns: array of Fock state occupation probabilities
    """
    n = np.arange(n_max + 1, dtype=int)

    r = np.abs(zeta)
    if r == 0:
        P_n = np.zeros_like(n)
        P_n[0] = 1
    else:
        P_n = (
            (np.tanh(r) ** (2 * n))
            / np.cosh(r)
            * np.exp(
                special.gammaln(2 * n + 1)
                - 2 * n * np.log(2)
                - 2 * special.gammaln(n + 1)
            )
        )
        # For a squeezed state, only even Fock states are occupied
        P_n[1::2] = 0

    return P_n


# pytype: enable=invalid-annotation
