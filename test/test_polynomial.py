import numpy as np
from test.common import TestBase
from fits import models


class TestLine(TestBase):
    """Tests for polynomials.Line

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """

    def setUp(self):
        super().setUp(model=models.Line())

    def test_line(self):
        x = np.linspace(-10, 10)
        params = {"a": 3.2, "y0": -9}
        self.check_single(
            x,
            params,
        )


class TestParabola(TestBase):
    """Tests for polynomials.Parabola

    This fitting code is already well covered by the tests on polynomials.Polynomial,
    here we just want to check for obvious mistakes in the wrapping code.
    """

    def setUp(self):
        super().setUp(model=models.Parabola(), plot_failures=True)

    def test_parabola(self):
        x = np.linspace(-10, 10)
        params = {"k": -9, "y0": +4, "x0": -3}
        self.check_single(x, params)


class TestPower(TestBase):
    """Tests for polynomials.Power"""

    def setUp(self):
        super().setUp(model=models.Power())

    def test_n(self):
        """Tests with `n` and `y0` floated for parameter sets with various values of
        `n` and `y0`. `x0` and `a` are held at single values.
        """
        x = np.linspace(0.1, 1)
        fixed_params = {"x0": 0, "a": 1}
        self.check_multiple(
            x,
            fixed_params,
            scanned_params={"n": [-5.51, -1, 1, 0, 3], "y0": [0, 1e-5, -1e-3]},
        )

    def test_a(self):
        """Tests with `a` and `y0` floated for parameter sets with various values of
        `n` and `y0`. `x0` and `n` are held at single values.
        """
        x = np.linspace(0.1, 15)
        fixed_params = {"x0": 0, "n": 5}
        self.check_multiple(
            x,
            fixed_params,
            scanned_params={"a": [0.5, 1, 3, 5.5], "y0": [-2.4, 0, 1, 100]},
        )

    def fuzz(self, num_trials=100, stop_at_failure=True, plot_failures=False):
        x = np.linspace(0.1, 15)
        static_params = {"x0": 0, "a": 1}
        return super().fuzz(
            x,
            static_params,
            fuzzed_params={"n": (-3, 3), "y0": (-10, 10)},
            num_trials=num_trials,
            stop_at_failure=stop_at_failure,
            plot_failures=plot_failures,
        )


class TestPolynomial(TestBase):
    """Tests for polynomials.Polynomial"""

    def setUp(self):
        super().setUp(model=models.Polynomial(), plot_failures=True)

    def test_polynomial(self):
        x = np.linspace(-5, 5)
        fixed_params = {"x0": 0}
        fixed_params.update({f"a_{n}": 0 for n in range(4, 11)})

        self.check_multiple(
            x,
            fixed_params,
            scanned_params={"a_0": [1], "a_1": [50], "a_2": [10], "a_3": [2]},
        )

    def test_x0(self):
        x = np.linspace(-5, 50) * 1e-3
        fixed_params = {f"a_{n}": 0 for n in range(3, 11)}

        # Floating x0 leads to an under-defined problem so we check the residuals rather
        # than the parameter values.
        self.check_multiple(
            x,
            fixed_params,
            scanned_params={"x0": [-10], "a_0": [1], "a_1": [50], "a_2": [10]},
            param_tol=None,
            residual_tol=1e-6,
        )

    def fuzz(self, num_trials=100, stop_at_failure=True, plot_failures=False):
        x = np.linspace(-5, 50) * 1e-3
        static_params = {f"a_{n}": 0 for n in range(4, 11)}
        static_params.update({"x0": 0})
        fuzzed_params = {f"a_{n}": (-10, 10) for n in range(4)}
        return super().fuzz(
            x,
            static_params,
            fuzzed_params=fuzzed_params,
            num_trials=num_trials,
            stop_at_failure=stop_at_failure,
            plot_failures=plot_failures,
        )
