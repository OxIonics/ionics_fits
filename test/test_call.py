import numpy as np

from ionics_fits.models.sinusoid import Sinusoid


def test_call(plot_failures):
    """Test that we can use the model as a callable"""
    x = np.linspace(0, 1, 100) * 2 * np.pi
    sinusoid = Sinusoid()
    sinusoid(x=x, a=1, omega=2, phi=0, y0=0)
