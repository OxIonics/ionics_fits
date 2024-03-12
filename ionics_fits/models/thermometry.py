import numpy as np
from typing import List, Tuple

from .polynomial import Line
from ..common import Model, ModelParameter, TX, TY
from ..normal import NormalFitter
from ..utils import scale_invariant, scale_x


class SidebandHeatingRate(Model):
    """Heating rate measured using sideband thermometry.

    This model calculates transition probabilities on the first red and blue motional
    sidebands of one or more ion(s) coupled to a single motional mode, whose mean
    occupancy is given by::

        n_bar(t) = n_bar_0 + n_bar_dot * t

    It assumes that:

    * the sidebands are both driven resonantly (zero detuning)
    * there are no state preparation or measurement errors
    * the driving field is applied for exactly a pi-pulse
    * the ion(s) start out entirely in the ground spin state
    * dephasing of the motion and spin coherence is negligible

    """

    def __init__(
        self,
        num_ions: int = 1,
        invert_r: bool = False,
        invert_b: bool = False,
        n_max: int = 30,
    ):
        """
        :param num_ions: number of ions coupled to the motional mode
        :param n_max: maximum Fock state to include in the simulation
        :param invert_r: if ``True`` we calculate the probability of the ion(s) not
          undergoing a transition on the red sideband. This is useful when fitting
          datasets where the un-excited RSB starts with a population of ``1`` not ``0``.
        :param invert_b: if ``True`` we calculate the probability of the ion(s) not
          undergoing a transition on the blue sideband. This is useful when fitting
          datasets where the un-excited BSB starts with a population of ``1`` not ``0``.
        """
        self.num_ions = num_ions
        self.invert_r = invert_r
        self.invert_b = invert_b
        self.n_max = n_max

        if self.num_ions not in [1, 2]:
            raise ValueError(f"Only 1 or 2 ions are supported, not {num_ions}")

        super().__init__()

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return [2, 6][self.num_ions - 1]

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [False] * self.get_num_y_axes()

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: TX,
        n_bar_0: ModelParameter(lower_bound=0, scale_func=scale_invariant),
        n_bar_dot: ModelParameter(lower_bound=0, scale_func=scale_x()),
    ) -> TY:
        """
        :param x: heating time
        :param n_bar_0: initial temperature of the motional mode
        :param n_bar_dot: heating rate (``quanta / s``)
        :returns: array of model values
        """
        x = np.atleast_1d(np.squeeze(x))

        n_vec = np.arange(self.n_max + 1)
        n_bar_vec = n_bar_0 + x * n_bar_dot

        n_bar, n = np.meshgrid(n_bar_vec, n_vec, indexing="ij")

        p_thermal = np.power(n_bar / (1 + n_bar), n) / (1 + n_bar)

        omega_eff_r = np.sqrt(n_vec)
        omega_eff_b = np.sqrt(n_vec + 1)

        P_r = np.sum(0.5 * (1 - np.cos(np.pi * omega_eff_r)) * p_thermal, axis=1)
        P_b = np.sum(0.5 * (1 - np.cos(np.pi * omega_eff_b)) * p_thermal, axis=1)

        P_r = 1 - P_r if self.invert_r else P_r
        P_b = 1 - P_b if self.invert_b else P_b

        return np.vstack((P_r, P_b))

    # pytype: enable=invalid-annotation

    def estimate_parameters(self, x: TX, y: TY):
        P_r = y[0, :]
        P_b = y[1, :]

        P_r = 1 - P_r if self.invert_r else P_r
        P_b = 1 - P_b if self.invert_b else P_b

        R = np.divide(P_r, P_b, out=np.zeros_like(P_r), where=P_b < 1)
        n_bar = np.divide(R, (1 - R), out=np.empty(P_r.shape), where=R < 1)

        fit = NormalFitter(x=x, y=n_bar, model=Line())

        self.parameters["n_bar_dot"].heuristic = fit.values["a"]
        self.parameters["n_bar_0"].heuristic = fit.values["y0"]
