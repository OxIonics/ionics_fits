import numpy as np
from common import TestBase
from fits import models


class TestPower(TestBase):
    """Tests for polynomials.Power"""

    def setUp(self):
        super().setUp(model_class=models.Power, plot_failures=True)

    def test_n(self):
        x = np.linspace(0.1, 50)
        fixed_params = {"x0": 0, "a": 5}
        self._test_multiple(
            x, fixed_params, scanned_params={"n": [0.5, 1, 3, 5.5], "y0": [0, 1, 100]}
        )

    def test_a(self):
        x = np.linspace(0.1, 50)
        fixed_params = {"x0": 0, "n": 5}
        self._test_multiple(
            x, fixed_params, scanned_params={"a": [0.5, 1, 3, 5.5], "y0": [0, 1, 100]}
        )

    def fuzz(self, num_trials=100):
        x = np.linspace(0.1, 50)
        fixed_params = {"x0": 0, "y0": 0}
        super().fuzz(
            x,
            fixed_params,
            fuzzed_params={"a": (0, 10), "n": (-3, 3)},
            num_trials=num_trials,
        )


class TestPolynomial(TestBase):
    """Tests for polynomials.Polynomial"""

    def setUp(self):
        super().setUp(model_class=models.Polynomial, plot_failures=True)

    def test_polynomial(self):
        x = np.linspace(-5, 50) * 1e-3
        fixed_params = {"x0": 0}
        fixed_params.update({f"a_{n}": 0 for n in range(4, 11)})

        self._test_multiple(
            x,
            fixed_params,
            scanned_params={"a_0": [1], "a_1": [50], "a_2": [10], "a_3": [2]},
        )

    def test_x0(self):
        x = np.linspace(-5, 50) * 1e-3
        fixed_params = {f"a_{n}": 0 for n in range(3, 11)}

        # Floating x0 we get a rather under-defined problem so we check the residuals
        # rather than the parameter values.
        self._test_multiple(
            x,
            fixed_params,
            scanned_params={"x0": [-10], "a_0": [1], "a_1": [50], "a_2": [10]},
            param_tol=None,
            residual_tol=1e-6,
        )

    def fuzz(self, num_trials=100):
        x = np.linspace(-5, 50) * 1e-3
        fixed_params = {f"a_{n}": 0 for n in range(4, 11)}
        fixed_params.update({"x0": 0})
        fuzzed_params = {f"a_{n}": (-10, 10) for n in range(4)}
        super().fuzz(
            x, fixed_params, fuzzed_params=fuzzed_params, num_trials=num_trials
        )
