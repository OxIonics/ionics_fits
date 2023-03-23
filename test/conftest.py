import matplotlib
import pytest

from . import common


def pytest_addoption(parser):
    parser.addoption(
        "--plot-failures",
        default=False,
        action="store_true",
        help="Show diagnostic plots from failed tests",
    )


@pytest.fixture
def plot_failures(request, monkeypatch):
    show = request.config.getoption("--plot-failures")
    if show:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Qt5Agg")
    else:
        matplotlib.use("agg")
        monkeypatch.setattr(common.plt, "show", lambda: None)
    return show
