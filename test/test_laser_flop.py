import numpy as np
import ionics_fits as fits
import test


def test_laser_flop():
    """Test for laser_flop.LaserFlop"""

    t = np.linspace(0, 5e-6)
    params = {
        "omega_0": 1e6 * 2 * np.pi,
        "n_bar": 0,
        "eta": 0.1,
        "P_readout_e": 1,
        "P_readout_g": 0,
        "t_dead": 0,
        "delta": 0,
    }

    model = fits.models.LaserFlopTimeThermal(sideband=0, n_max=100)
    model.parameters["omega_0"].fixed_to = params["omega_0"]
    model.parameters["eta"].fixed_to = params["eta"]
    model.parameters["n_bar"].initialised_to = 10
    test.common.check_multiple_param_sets(
        t,
        model,
        params,
        test.common.TestConfig(plot_failures=True, plot_all=True),
    )
