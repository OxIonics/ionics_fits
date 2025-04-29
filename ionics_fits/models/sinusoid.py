from typing import Dict, List, Tuple

import numpy as np

from ..common import TX, TY, Model, ModelParameter
from ..utils import scale_invariant, scale_x, scale_x_inv, scale_y
from . import heuristics, utils
from .transformations.reparametrized_model import ReparametrizedModel


class Sinusoid(Model):
    """Generalised sinusoid fit according to::

        y = Gamma * a * sin[omega * (x - x0) + phi] + y0
        Gamma = exp(-x / tau).

    All phases are in radians, frequencies are in angular units.

    ``x0`` and ``phi0`` are equivalent parametrisations for the phase offset, but in
    some cases it works out convenient to have access to both (e.g. one as a fixed
    offset, the other floated). At most one of them should be floated at once. By
    default, ``x0`` is fixed at 0 and ``phi0`` is floated.
    """

    def get_num_x_axes(self) -> int:
        return 1

    def get_num_y_axes(self) -> int:
        return 1

    def can_rescale(self) -> Tuple[List[bool], List[bool]]:
        return [True], [True]

    # pytype: disable=invalid-annotation,signature-mismatch
    def _func(
        self,
        x: TX,
        a: ModelParameter(lower_bound=0, scale_func=scale_y()),
        omega: ModelParameter(lower_bound=0, scale_func=scale_x_inv()),
        phi: utils.PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
            scale_func=scale_invariant,
        ),
        y0: ModelParameter(scale_func=scale_y()),
        x0: ModelParameter(fixed_to=0, scale_func=scale_x()),
        tau: ModelParameter(
            lower_bound=0,
            fixed_to=np.inf,
            scale_func=scale_x(),
        ),
    ) -> TY:
        """
        :param a: initial (``x = 0``) amplitude of the sinusoid
        :param omega: angular frequency
        :param phi: phase offset
        :param y0: y-axis offset
        :param x0: x-axis offset
        :param tau: decay/growth constant
        """
        Gamma = np.exp(-x / tau)
        return Gamma * a * np.sin(omega * (x - x0) + phi) + y0

    # pytype: enable=invalid-annotation,signature-mismatch

    def estimate_parameters(self, x: TX, y: TY):
        x = np.squeeze(x)
        y = np.squeeze(y)

        # We don't have good heuristics for these parameters
        self.parameters["y0"].heuristic = np.mean(y)
        self.parameters["tau"].heuristic = np.max(x)

        omega, spectrum = heuristics.get_pgram(x, y)
        peak = np.argmax(spectrum)

        self.parameters["a"].heuristic = spectrum[peak]
        self.parameters["omega"].heuristic = omega[peak]

        phi, _ = heuristics.param_min_sqrs(
            model=self,
            x=x,
            y=y,
            scanned_param="phi",
            scanned_param_values=np.linspace(-np.pi, np.pi, num=20),
            defaults={"x0": 0},
        )

        phi = self.parameters["phi"].clip(phi)

        if self.parameters["x0"].fixed_to is None:
            if self.parameters["phi"].fixed_to is None:
                raise ValueError("Only one of 'x0' and 'phi' may be floated at once")

            self.parameters["phi"].heuristic = 0
            self.parameters["x0"].heuristic = (
                -phi / self.parameters["omega"].get_initial_value()
            )
        else:
            self.parameters["phi"].heuristic = phi
            self.parameters["x0"].heuristic = 0.0

    def calculate_derived_params(
        self,
        x: TX,
        y: TY,
        fitted_params: Dict[str, float],
        fit_uncertainties: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        * ``f``: frequency
        * ``phi_cosine``: cosine phase (``phi + pi/2``)
        * ``contrast``: peak-to-peak amplitude of the pure sinusoid
        * ``min``/``max``: min / max values of the pure sinusoid
        * ``period``: period of oscillation

        TODO: peak values of the damped sinusoid as well as ``x`` value that the peak
        occurs at.
        """
        derived_params = {}
        derived_params["f"] = fitted_params["omega"] / (2 * np.pi)
        derived_params["phi_cosine"] = fitted_params["phi"] + np.pi / 2
        derived_params["contrast"] = 2 * np.abs(fitted_params["a"])
        derived_params["min"] = fitted_params["y0"] - np.abs(fitted_params["a"])
        derived_params["max"] = fitted_params["y0"] + np.abs(fitted_params["a"])
        derived_params["period"] = 2 * np.pi / fitted_params["omega"]

        derived_uncertainties = {}
        derived_uncertainties["f"] = fit_uncertainties["omega"] / (2 * np.pi)
        derived_uncertainties["phi_cosine"] = fit_uncertainties["phi"]
        derived_uncertainties["contrast"] = 2 * fit_uncertainties["a"]
        derived_uncertainties["min"] = np.sqrt(
            fit_uncertainties["y0"] ** 2 + fit_uncertainties["a"] ** 2
        )
        derived_uncertainties["max"] = np.sqrt(
            fit_uncertainties["y0"] ** 2 + fit_uncertainties["a"] ** 2
        )
        derived_uncertainties["period"] = (
            2 * np.pi * fit_uncertainties["omega"] / (fitted_params["omega"] ** 2)
        )

        return derived_params, derived_uncertainties


class SineMinMax(ReparametrizedModel):
    """Sinusoid parametrised by minimum / maximum values instead of offset / amplitude::

            y = Gamma * a * sin[omega * (x - x0) + phi] + y0

    This class is equivalent to :class:`Sinusoid` except that the ``a`` and ``y0``
    parameters are replaced with new ``min`` and ``max`` parameters defined by::

      min = y0 - a
      max = y0 + a

    See :class:`Sinusoid` for further details.
    """

    def __init__(self):
        super().__init__(
            model=Sinusoid(),
            new_params={
                "min": ModelParameter(scale_func=scale_y()),
                "max": ModelParameter(scale_func=scale_y()),
            },
            bound_params=["a", "y0"],
        )

    @staticmethod
    def bound_param_values(param_values: Dict[str, float]) -> Dict[str, float]:
        return {
            "a": 0.5 * (param_values["max"] - param_values["min"]),
            "y0": 0.5 * (param_values["max"] + param_values["min"]),
        }

    @staticmethod
    def bound_param_uncertainties(
        param_values: Dict[str, float], param_uncertainties: Dict[str, float]
    ) -> Dict[str, float]:
        err = 0.5 * np.sqrt(
            param_uncertainties["max"] ** 2 + param_uncertainties["min"] ** 2
        )
        return {"a": err, "y0": err}

    @staticmethod
    def new_param_values(model_param_values: Dict[str, float]) -> Dict[str, float]:
        return {
            "max": (model_param_values["y0"] + model_param_values["a"]),
            "min": (model_param_values["y0"] - model_param_values["a"]),
        }


class Sine2(Sinusoid):
    """Sine-squared fit according to::

        y = Gamma * a * [sin(omega * (x - x0) + phi)]**2 + y0
        Gamma = np.exp(-x / tau)

    See also :class:`~ionics_fits.models.sinusoid.Sinusoid`.
    """

    # pytype: disable=invalid-annotation
    def _func(
        self,
        x: TX,
        a: ModelParameter(lower_bound=0, scale_func=scale_y()),
        omega: ModelParameter(lower_bound=0, scale_func=scale_x_inv()),
        phi: utils.PeriodicModelParameter(
            period=2 * np.pi,
            offset=-np.pi,
            scale_func=scale_invariant,
        ),
        y0: ModelParameter(scale_func=scale_y()),
        x0: ModelParameter(fixed_to=0, scale_func=scale_x()),
        tau: ModelParameter(
            lower_bound=0,
            fixed_to=np.inf,
            scale_func=scale_x(),
        ),
    ) -> TY:
        """
        :param a: initial (``x = 0``) amplitude of the sinusoid
        :param omega: angular frequency
        :param phi: phase offset
        :param y0: y-axis offset
        :param x0: x-axis offset
        :param tau: decay/growth constant
        """
        Gamma = np.exp(-x / tau)
        return Gamma * a * (np.sin(omega * (x - x0) + phi) ** 2) + y0

    # pytype: enable=invalid-annotation

    def estimate_parameters(self, x: TX, y: TY):
        # Use the identity: sin(x)**2 = 1/2 * (1 + sin(2*x- pi/2))
        # a * [sin(omega * (x - x0) + phi)]**2 + y0
        # = 0.5 * a * (1 + sin(2*omega * (x - x0) + 2*phi - pi/2)) + y0
        # = a' * sin(omega' * (x - x0) + phi') + y0'
        #
        # where
        #  - a' = 0.5 * a
        #  - omega' = 2 * omega
        #  - phi' = 2 * phi - pi / 2
        #  - y0' = y0 + a'

        sine = Sinusoid()

        phi_attrs = ("fixed_to", "user_estimate")
        attrs = (*phi_attrs, "lower_bound", "upper_bound")

        for param in ("x0", "tau"):
            for attr_name in attrs:
                attr_value = getattr(self.parameters[param], attr_name)
                setattr(sine.parameters[param], attr_name, attr_value)

        for attr_name in attrs:
            attr_value = getattr(self.parameters["a"], attr_name)
            if attr_value is None:
                attr_value_pr = None
            else:
                attr_value_pr = 0.5 * attr_value
            setattr(sine.parameters["a"], attr_name, attr_value_pr)

        for attr_name in attrs:
            attr_value = getattr(self.parameters["omega"], attr_name)
            if attr_value is None:
                attr_value_pr = None
            else:
                attr_value_pr = 2 * attr_value
            setattr(sine.parameters["omega"], attr_name, attr_value_pr)

        for attr_name in phi_attrs:
            attr_value = getattr(self.parameters["phi"], attr_name)
            if attr_value is None:
                attr_value_pr = None
            else:
                attr_value_pr = 2 * attr_value - np.pi / 2
            setattr(sine.parameters["phi"], attr_name, attr_value_pr)

        # y0 and a are coupled in the sine2x model so we can't trivially transfer
        # bounds across
        if (
            self.parameters["y0"].has_user_initial_value()
            and sine.parameters["a"].has_user_initial_value()
        ):
            a_pr = sine.parameters["a"].get_initial_value()
            y0 = self.parameters["y0"].get_initial_value()
            y0_pr = y0 + a_pr
            sine.parameters["y0"].user_estimate = y0_pr

        sine.estimate_parameters(x, y)

        self.parameters["tau"].heuristic = sine.parameters["tau"].get_initial_value()
        self.parameters["x0"].heuristic = sine.parameters["x0"].get_initial_value()

        a_pr = sine.parameters["a"].get_initial_value()
        omega_pr = sine.parameters["omega"].get_initial_value()
        phi_pr = sine.parameters["phi"].get_initial_value()
        y0_pr = sine.parameters["y0"].get_initial_value()

        self.parameters["a"].heuristic = 2 * a_pr
        self.parameters["omega"].heuristic = omega_pr / 2
        self.parameters["phi"].heuristic = (phi_pr + np.pi / 2) / 2
        self.parameters["y0"].heuristic = y0_pr - a_pr
