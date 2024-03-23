import numpy as np
from typing import Callable, Dict, List, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from .common import Model, TX

    num_x_axes = float
    num_y_axes = float


class _SubscriptableNumpyArray(type):
    def __getitem__(cls, _):
        return np.ndarray


class Array(np.ndarray, metaclass=_SubscriptableNumpyArray):
    """Subclass of numpy's NDArray used purely for type annotation convenience.

    Type annotations can have arbitrary subscripts, e.g. ``Array[(4,), "float64"]``.
    """


class _SubscriptableArrayLikeUnion(type):
    def __getitem__(cls, x):
        return Union[List, Array[x]]


class ArrayLike(metaclass=_SubscriptableArrayLikeUnion):
    """
    Used for types that can be a numpy array or a list that's then converted to an array
    """


TX_SCALE = Array[("num_x_axes",), np.float64]
TY_SCALE = Array[("num_y_axes",), np.float64]
TSCALE_FUN = Callable[[TX_SCALE, TY_SCALE], float]


def scale_invariant(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
    r"""Scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose value is
    invariant under rescaling of the x- and y-axes

    :param x_scales: array of x-axis scale factors
    :param y_scales: array of y-axis scale factors
    :returns: ``1``
    """
    return 1


def scale_x(x_axis: int = 0) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose
    value scales linearly with one x-axis dimension.

    :param x_axis: index of the x-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return x_scales[x_axis]

    fun.__name__ = "scale_x"
    return fun


def scale_x_inv(x_axis: int = 0) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose
    value scales inversely with one x-axis dimension.

    :param x_axis: index of the x-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return 1 / x_scales[x_axis]

    fun.__name__ = "scale_x_inv"
    return fun


def scale_y(y_axis: int = 0) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose
    value scales linearly with one y-axis dimension.

    :param y_axis: index of the y-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return y_scales[y_axis]

    fun.__name__ = "scale_y"
    return fun


def scale_power(
    x_power: int, y_power: int, x_axis: int = 0, y_axis: int = 0
) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose
    value scales as a function of one x-axis and one y-axis dimension.

    The parameter scale factor is calculated as::

        scale_factor = (x_scales[x_axis] ** x_power) * (y_scales[y_axis] ** y_power)

    :param x_power: x-axis power
    :param y_power: y-axis power
    :param x_axis: index of the x-axis dimension the parameter scales with
    :param y_axis: index of the y-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return (x_scales[x_axis] ** x_power) * (y_scales[y_axis] ** y_power)

    fun.__name__ = "scale_power"
    return fun


def scale_undefined(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
    r"""Scale function for :class:`~ionics_fits.common.ModelParameter`\ s whose scaling
    is not known yet.

    This scale function is typically used for parameters whose scale factor is not known
    until runtime.

    :param x_scales: array of x-axis scale factors
    :param y_scales: array of y-axis scale factors
    """

    raise RuntimeError(
        "Attempt to rescale model parameter with undefined scale function"
    )


def scale_no_rescale(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
    r"""Scale function for :class:`~ionics_fits.common.ModelParameter`\ s which cannot be
    rescaled.

    Raises a ``RuntimeError`` if any of the x-axis or y-axis scale factors are not equal
    to ``1``.

    :param x_scales: array of x-axis scale factors
    :param y_scales: array of y-axis scale factors
    :returns: ``1``.
    """
    if any([x_scale != 1.0 for x_scale in x_scales]):
        raise RuntimeError(
            "Attempt to rescale model parameter along x which does not support "
            "rescaling"
        )
    if any([y_scale != 1.0 for y_scale in y_scales]):
        raise RuntimeError(
            "Attempt to rescale model parameter along y which does not support "
            "rescaling"
        )

    return 1.0


def step_param(
    model: "Model",
    stepped_param: str,
    param_values: Dict[str, float],
    step_size: float = 1e-4,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Generates parameter value dictionaries with one parameter stepped. Used by
    numerical differentiation routines.

    :param model: :class:`~ionics_fits.common.Model` whose parameters we are varying
    :param stepped_param: name of the stepped parameter
    :param param_values: dictionary of parameter values
    :param step_size: base step size. As a rule of thumb this should not be larger than
        ``~1e-5``. See section 5.7 of `Numerical Recipes` (third edition) for discussion
        of numerical differentiation. If the stepped parameter has lower and upper
        bounds set, we use ``(upper_bound - lower_bound) * step_size`` as the step size,
        otherwise we use ``step_size`` directly.
    :returns: tuple of ``(lower_values, upper_values, step)`` where: ``lower_values`` is
        a modified ``param_values`` dictionary with ``stepped_param`` decreased by half
        of the step size (clipped to its lower bound); ``upper_values`` is a modified
        ``param_values`` dictionary with ``stepped_param`` stepped upwards; and,
        ``step`` is the size of the step taking clipping into account.
    """
    param_data = model.parameters[stepped_param]
    lower_bound = param_data.lower_bound
    upper_bound = param_data.upper_bound
    stepped_param_value = param_values[stepped_param]

    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        step_size = step_size * (upper_bound - lower_bound)
    else:
        step_size = step_size

    param_lower = np.clip(
        stepped_param_value - 0.5 * step_size,
        a_min=lower_bound,
        a_max=upper_bound,
    )
    param_upper = np.clip(
        stepped_param_value + 0.5 * step_size,
        a_min=lower_bound,
        a_max=upper_bound,
    )

    step = param_upper - param_lower

    lower_values = dict(param_values)
    upper_values = dict(param_values)
    lower_values[stepped_param] = param_lower
    upper_values[stepped_param] = param_upper

    return lower_values, upper_values, step


def num_diff(
    model: "Model",
    diff_param: str,
    x: "TX",
    param_values: Dict[str, float],
    step_size: float = 1e-4,
) -> float:
    """Returns the numerical derivative of a model function with respect to one of its
    parameter.

    This function calculates the derivative::

        d(fun)/d(param)[x, param_values]

    :param model: the model function to differentiate
    :param diff_param: name of the parameter to differentiate the model function with
        respect to
    :param x: x-axis point to evaluate the derivative at
    :param param_values: dictionary of parameter values to evaluate the derivative at.
    :param step_size: base step size. As a rule of thumb this should not be larger than
        ``~1e-5``. See section 5.7 of `Numerical Recipes` (third edition) for discussion
        of numerical differentiation. If the stepped parameter has lower and upper
        bounds set, we use ``(upper_bound - lower_bound) * step_size`` as the step size,
        otherwise we use ``step_size`` directly.
    :returns: the model function's derivative with respect to the selected parameter
    """
    lower_values, upper_values, param_step = step_param(
        model=model,
        stepped_param=diff_param,
        param_values=param_values,
        step_size=step_size,
    )

    f_lower = model.func(x, lower_values)
    f_upper = model.func(x, upper_values)

    return (f_upper - f_lower) / param_step
