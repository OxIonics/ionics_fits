from typing import cast, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy import fft, signal

from .. import Model
from ..utils import Array, ArrayLike


if TYPE_CHECKING:
    num_samples = float
    num_y_channels = float
    num_values = float
    num_spectrum_pts = float
    num_spectrum_samples = float


def param_min_sqrs(
    model: Model,
    x: Array[("num_samples",), np.float64],
    y: Array[("num_y_channels", "num_samples"), np.float64],
    scanned_param: str,
    scanned_param_values: ArrayLike["num_values", np.float64],
    defaults: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """Scans one model parameter while holding the others fixed to find the value
    that gives the best fit to the data (minimum sum-squared residuals).

    :param x: x-axis data
    :param y: y-axis data
    :param scanned_param: name of parameter to optimize
    :param scanned_param_values: array of scanned parameter values to test
    :param defaults: optional dictionary of fallback values to use for non-scanned
      parameters which don't have heuristics set yet

    :returns: tuple with the value from :param scanned_param_values: which results
    in lowest residuals and the root-sum-squared residuals for that value.
    """
    defaults = cast(dict, defaults or {})
    if not set(defaults.keys()).issubset(model.parameters.keys()):
        raise ValueError("Defaults must be a subset of the model parameters")
    if scanned_param in defaults.keys():
        raise ValueError("The scanned parameter cannot have a default")

    missing_values = [
        param_name
        for param_name, param_data in model.parameters.items()
        if not param_data.has_initial_value()
        and param_name != scanned_param
        and param_name not in defaults.keys()
    ]
    if any(missing_values):
        raise ValueError(f"No initial value specified for parameters: {missing_values}")

    param_values = {
        param: param_data.get_initial_value(default=defaults.get(param))
        for param, param_data in model.parameters.items()
        if param != scanned_param
    }

    scanned_param_values = np.asarray(scanned_param_values).squeeze()
    costs = np.zeros_like(scanned_param_values)
    for idx, value in np.ndenumerate(scanned_param_values):
        param_values[scanned_param] = value
        y_params = model.func(x, param_values)
        costs[idx] = np.sqrt(np.sum(np.square(y - y_params)))

    # handle a quirk of numpy indexing if only one value is passed in
    if scanned_param_values.size == 1:
        return float(scanned_param_values), float(costs)

    opt = np.argmin(costs)
    return float(scanned_param_values[opt]), float(costs[opt])


def get_sym_x(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_y_channels", "num_samples"), np.float64],
) -> float:
    """Returns `x_0` such that y(x-x_0) is maximally symmetric."""
    y = np.atleast_2d(y)
    x_span = x.ptp()
    num_samples = x.size
    window_min = min(x) + 0.125 * x_span
    window_max = max(x) - 0.125 * x_span

    # min of three points per window
    window_min = max(window_min, x[2])
    window_max = min(window_max, x[-3])

    window = np.argwhere(np.logical_and(x >= window_min, x <= window_max)).squeeze()
    costs = np.zeros_like(window, dtype=float)

    for point_num, idx in np.ndenumerate(window.squeeze()):
        samples_left = idx
        samples_right = num_samples - idx - 1
        num_samples_idx = min(samples_left, samples_right)
        idx_left = idx - num_samples_idx
        idx_right = idx + num_samples_idx

        samples_left = y[:, idx_left:idx]
        samples_right = y[:, (idx + 1) : (idx_right + 1)]

        assert samples_left.shape == samples_right.shape

        diff = samples_left - np.fliplr(samples_right)

        # give more weight to windows with more structure
        mu = np.mean(samples_left)
        var = np.sum(np.square(samples_left - mu)) / samples_left.size

        cost = np.sqrt(np.sum(np.square(diff))) / samples_left.size
        costs[point_num] = cost / var

    return x[window[np.argmin(costs)]]


def find_x_offset_fft(
    x: Array[("num_samples",), np.float64],
    omega: Array[("num_spectrum_pts",), np.float64],
    spectrum: Array[("num_spectrum_pts",), np.float64],
    omega_cut_off: float,
) -> float:
    """Finds the x-axis offset of a dataset from the phase of an FFT.

    This function uses the FFT shift theorem to extract the offset from the phase
    slope of an FFT. At present it only supports models with a single y channel.

    :param omega: FFT frequency axis
    :param spectrum: complex FFT data. For models with multiple y channels, this
      should contain data from a single channel only.
    :param omega_cut_off: highest value of omega to use in offset estimation

    :returns: an estimate of the x-axis offset
    """
    if spectrum.ndim != 1:
        raise ValueError(f"Function only takes 1 y channel, not {spectrum.shape[1]}")

    keep = omega < omega_cut_off
    if np.sum(keep) < 2:
        raise ValueError("Insufficient data below cut-off")

    omega = omega[keep]
    phi = np.unwrap(np.angle(spectrum[keep]))
    phi -= phi[0]

    p = np.polyfit(omega, phi, deg=1)

    x0 = min(x) - p[0]
    x0 = x0 if x0 > min(x) else x0 + x.ptp()
    return x0


def find_x_offset_sym_peak_fft(
    model: Model,
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
    omega: Array[("num_spectrum_pts",), np.float64],
    spectrum: Array[("num_spectrum_pts",), np.float64],
    omega_cut_off: float,
    test_pts: Optional[Array[("num_values",), np.float64]] = None,
    x_offset_param_name: str = "x0",
    y_offset_param_name: str = "y0",
    defaults: Optional[Dict[str, float]] = None,
):
    """Finds the x-axis offset for symmetric, peaked (maximum deviation from the
    baseline occurs at the origin) functions.

    This heuristic draws candidate x-offset points from three sources and picks the
    best one (in the least-squares residuals sense). Sources:
      - FFT shift theorem based on provided spectrum data
      - Tests all points in the top quartile of deviation from the baseline
      - Optionally, user-provided "test points", taken from another heuristic. This
        allows the developer to combine the general-purpose heuristics here with
        other heuristics which make use of more model-specific assumptions

    :param model: the fit model
    :param x: x-axis data
    :param y: y-axis data. For models with multiple y channels, this should contain
        data from a single channel only.
    :param omega: FFT frequency axis
    :param spectrum: complex FFT data. For models with multiple y channels, this
      should contain data from a single channel only.
    :param omega_cut_off: highest value of omega to use in offset estimation
    :param test_pts: optional array of x-axis points to test
    :param x_offset_param_name: name of the x-axis offset model parameter
    :param y_offset_param_name: name of the y-axis offset model parameter
    :param defaults: optional dictionary of fallback values to use for parameters with
        no initial value specified.

    :returns: an estimate of the x-axis offset
    """
    defaults = defaults or {}

    if y.ndim != 1:
        raise ValueError(
            f"{y.shape[0]} y-channels were provided to a method which takes 1."
        )

    x0_candidates = np.array([])

    if test_pts is not None:
        x0_candidates = np.append(x0_candidates, test_pts)

    try:
        fft_candidate = find_x_offset_fft(
            x=x, omega=omega, spectrum=spectrum, omega_cut_off=omega_cut_off
        )
        x0_candidates = np.append(x0_candidates, fft_candidate)
    except ValueError:
        pass

    y0_default = defaults.get(y_offset_param_name)
    y0 = model.parameters[y_offset_param_name].get_initial_value(y0_default)
    deviations = np.argsort(np.abs(y - y0))
    top_quartile_deviations = deviations[int(len(deviations) * 3 / 4) :]
    deviations_candidates = x[top_quartile_deviations]
    x0_candidates = np.append(x0_candidates, deviations_candidates)

    best_x0, _ = param_min_sqrs(
        model=model,
        x=x,
        y=y,
        scanned_param=x_offset_param_name,
        scanned_param_values=x0_candidates,
        defaults=defaults,
    )
    return best_x0


def get_spectrum(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
    trim_dc: bool = False,
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns the frequency spectrum (Fourier transform) of a dataset.
    
    NB the returned spectrum will only match the Fourier transform of the model function
    in the limit where the model function is zero outside of the sampling window.

    NB for narrow-band signals the peak amplitude depends on where the signal frequency
    lies compared to the frequency bins.

    :param x: 1D ndarray of shape (num_samples,) containing x-axis data
    :param y: 1D ndarray of shape (num_samples,) containing y-axis data
    :param trim_dc: if `True` we do not return the DC component.
    :returns: tuple of (angular freq, fft)
    """
    if x.ndim != 1:
        raise ValueError("x-axis data must be a 1D array.")
    if y.ndim != 1:
        raise ValueError("y-axis data must be a 1D array.")

    dx = x.ptp() / x.size
    n = x.size
    omega = np.fft.fftfreq(n, dx) * (2 * np.pi)
    y_f = fft.rfft(y) * dx

    y_f = y_f[: int(n / 2)]
    omega = omega[: int(n / 2)]

    if trim_dc:
        omega = omega[1:]
        y_f = y_f[1:]

    return omega, y_f


def get_pgram(
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples",), np.float64],
) -> Tuple[
    Array[("num_spectrum_samples",), np.float64],
    Array[("num_spectrum_samples",), np.float64],
]:
    """Returns a periodogram for a dataset, converted into amplitude units.

    Based on the Lombe-Scargle periodogram (essentially least-squares fitting of
    sinusoids at different frequencies).

    :param x: x-axis data
    :param y: y-axis data. For models with multiple y channels, this should contain
        data from a single channel only.
    :returns: tuple with the frequency axis (angular units) and the periodogram
    """
    if y.ndim != 1 and y.shape[1] > 1:
        raise ValueError(
            f"{y.shape[1]} y channels were provided to a method which takes 1"
        )

    dx = np.min(np.diff(x))
    duration = x.ptp()
    n = int(duration / dx)
    df = 1 / (n * dx)

    f_nyquist = 0.5 / dx

    omega_list = 2 * np.pi * np.linspace(df, f_nyquist, n)
    pgram = signal.lombscargle(x, y, omega_list, precenter=True)
    pgram = np.sqrt(np.abs(pgram) * 4 / len(y))

    return omega_list, pgram


def find_x_offset_sampling(
    model: Model,
    x: Array[("num_samples",), np.float64],
    y: Array[("num_samples", "num_y_channels"), np.float64],
    width: float,
    x_offset_param_name: str = "x0",
) -> float:
    """Finds the x-axis offset of a dataset by stepping through a range of potential
    offset values and picking the one that gives the lowest residuals.

    This function takes a more brute-force approach by evaluating the model at a
    range of offset values, picking the one that gives the lowest residuals. This
    may be appropriate where one needs the estimate to be highly robust in the face
    of noisy, irregularly sampled data.

    :param x: x-axis data
    :param y: y-axis data
    :param width: width of the feature we're trying to find (e.g. FWHMH). Used to
        pick the spacing between offset values to try.
    :param x_offset_param_name: name of the x-axis offset parameter

    :returns: an estimate of the x-axis offset
    """
    offsets = np.arange(min(x), max(x), width / 6)
    return param_min_sqrs(
        model=model,
        x=x,
        y=y,
        scanned_param=x_offset_param_name,
        scanned_param_values=offsets,
    )[0]
