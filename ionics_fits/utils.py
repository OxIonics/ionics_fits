from typing import TYPE_CHECKING, Callable, List, Union

import numpy as np

if TYPE_CHECKING:
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
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s
    whose value scales linearly with one x-axis dimension.

    :param x_axis: index of the x-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return x_scales[x_axis]

    fun.__name__ = "scale_x"
    return fun


def scale_x_inv(x_axis: int = 0) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s
    whose value scales inversely with one x-axis dimension.

    :param x_axis: index of the x-axis dimension the parameter scales with
    :returns: scale function
    """

    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        return 1 / x_scales[x_axis]

    fun.__name__ = "scale_x_inv"
    return fun


def scale_y(y_axis: int = 0) -> TSCALE_FUN:
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s
    whose value scales linearly with one y-axis dimension.

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
    r"""Returns a scale function for :class:`~ionics_fits.common.ModelParameter`\ s
    whose value scales as a function of one x-axis and one y-axis dimension.

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
    r"""Scale function for :class:`~ionics_fits.common.ModelParameter`\ s which cannot
    be rescaled.

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


def to_float(x) -> float:
    return float(x.item() if isinstance(x, np.ndarray) else x)
