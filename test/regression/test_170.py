import numpy as np

from ionics_fits.models.rabi import RabiFlop


def test_170(plot_failures):
    """Test that `func` works correctly for the `RabiFlop` base class"""
    # For now, we just test that these don't raise exceptions related to the model not
    # handling the data correctly
    model = RabiFlop(start_excited=False)
    params = {"P_readout_e": 1, "P_readout_g": 0, "omega": 1, "w_0": 0}

    t = 0
    delta = 0
    model((t, delta), **params)

    t = np.linspace(0, 1)
    delta = 0
    model((t, delta), **params)

    t = np.linspace(0, 1)
    delta = np.linspace(-1, 1)
    model((t, delta), **params)

    t = np.atleast_2d(np.linspace(0, 1))
    delta = np.linspace(-1, 1)
    model((t, delta), **params)

    t = np.atleast_2d(np.linspace(0, 1))
    delta = np.atleast_2d(np.linspace(-1, 1))
    model((t, delta), **params)
