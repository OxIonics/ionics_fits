import numpy as np

from ionics_fits.models.gaussian import Gaussian
from ionics_fits.models.rabi import RabiFlopFreq
from ionics_fits.models.transformations.repeated_model import RepeatedModel
from ionics_fits.normal import NormalFitter

from .common import Config, check_multiple_param_sets


class DoubleRabiFreq(RepeatedModel):
    def __init__(self):
        super().__init__(
            model=RabiFlopFreq(start_excited=True),
            common_params=[
                "P_readout_e",
                "P_readout_g",
                "t_pulse",
                "omega",
                "tau",
                "t_dead",
            ],
            num_repetitions=2,
        )


class QuadrupleRabiFreq(RepeatedModel):
    def __init__(self):
        super().__init__(
            model=DoubleRabiFreq(),
            common_params=[
                "P_readout_e",
                "P_readout_g",
                "t_pulse",
                "omega",
                "tau",
                "t_dead",
            ],
            num_repetitions=2,
        )


def test_repeated_model(plot_failures):
    """Test for transformations.RepeatedModel"""
    w = np.linspace(-10, 10, 200)
    params = {
        "w_0_0_0": [-3.0],
        "w_0_0_1": [+3.0],
        "w_0_1_0": [-1.0],
        "w_0_1_1": [+1.0],
        "P_readout_e": 1,
        "P_readout_g": 0,
        "t_pulse": 1,
        "omega": np.pi,
        "tau": np.inf,
        "t_dead": 0,
    }

    model = QuadrupleRabiFreq()
    model.parameters["P_readout_e"].fixed_to = params["P_readout_e"]
    model.parameters["P_readout_g"].fixed_to = params["P_readout_g"]
    model.parameters["t_pulse"].fixed_to = params["t_pulse"]

    y = np.atleast_2d(model.func(w, params))

    if y.shape != (4, len(w)):
        raise ValueError("Incorrect y-shape for repeated model.")

    check_multiple_param_sets(
        w,
        model,
        params,
        Config(plot_failures=plot_failures, param_tol=None, residual_tol=1e-8),
    )


def test_repeated_model_derived_params(plot_failures):
    """Test that RepeatedModel handles derived parameters correctly"""
    base_model = Gaussian()
    num_repetitions = 5

    x = np.linspace(-5, 5)
    params = {"x0": 0, "y0": 0, "a": 1, "sigma": 1}
    y = np.tile(np.squeeze(base_model(x, **params)), (num_repetitions, 1))

    # Case 1: don't aggregate results
    repeated_model = RepeatedModel(
        model=base_model,
        common_params=["x0", "y0", "a", "sigma"],
        num_repetitions=num_repetitions,
    )

    model_derived_results = ["FWHMH", "w0", "peak"]
    expected_derived_results = []
    for result in model_derived_results:
        expected_derived_results += [
            f"{result}_{idx}" for idx in range(num_repetitions)
        ]
        expected_derived_results += [f"{result}_mean", f"{result}_peak_peak"]

    fit = NormalFitter(x=x, y=y, model=repeated_model)

    assert len(fit.derived_values.keys()) == len(set(fit.derived_values.keys()))
    assert len(expected_derived_results) == len(expected_derived_results)
    assert set(fit.derived_values.keys()) == set(expected_derived_results)

    # Case 2: aggregate results
    repeated_model = RepeatedModel(
        model=base_model,
        common_params=["x0", "y0", "a", "sigma"],
        num_repetitions=num_repetitions,
        aggregate_results=True,
    )

    model_derived_results = ["FWHMH", "w0", "peak"]

    fit = NormalFitter(x=x, y=y, model=repeated_model)

    assert len(fit.derived_values.keys()) == len(set(fit.derived_values.keys()))
    assert len(model_derived_results) == len(model_derived_results)
    assert set(fit.derived_values.keys()) == set(model_derived_results)
