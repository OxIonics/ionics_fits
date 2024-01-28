import numpy as np
from typing import Callable, List, Union, TYPE_CHECKING


if TYPE_CHECKING:
    num_x_axes = float
    num_y_axes = float


class _SubscriptableNumpyArray(type):
    def __getitem__(cls, _):
        return np.ndarray


class Array(np.ndarray, metaclass=_SubscriptableNumpyArray):
    """
    Subclass of numpy's NDArray used purely for type annotation convenience
    Type annotations can have arbitrary subscripts, e.g. Array[(4,), "float64"].
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
    """Scale function for model parameters whose value is invariant under rescaling of
    the x- and y-axes"""
    return 1


def scale_x(x_axis: int = 0) -> TSCALE_FUN:
    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        """
        Scale function for model parameters whose value scales linearly with one x-axis
        dimension.
        """
        return x_scales[x_axis]

    return fun


def scale_x_inv(x_axis: int = 0) -> TSCALE_FUN:
    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        """
        Scale function for model parameters whose value scales inversely with one x-axis
        dimension.
        """
        return 1 / x_scales[x_axis]

    return fun


def scale_y(y_axis: int = 0) -> TSCALE_FUN:
    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        """
        Scale function for model parameters whose value scales linearly with one y-axis
        dimension.
        """
        return y_scales[y_axis]

    return fun


def scale_power(
    x_power: int, y_power: int, x_axis: int = 0, y_axis: int = 0
) -> TSCALE_FUN:
    def fun(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
        """
        Scale function for model parameters whose value scales as a power of one x-axis
        and one y-axis dimension.
        """
        return (x_scales[x_axis] ** x_power) * (y_scales[y_axis] ** y_power)

    return fun


def scale_undefined(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
    """This is typically used when the appropriate scale factor to use must be
    determined at runtime"""
    raise RuntimeError(
        "Attempt to rescale model parameter with undefined scale function"
    )


def scale_no_rescale(x_scales: TX_SCALE, y_scales: TY_SCALE) -> float:
    """For model parameters which cannot be rescaled"""
    if any(x_scales != 1.0) or any(y_scales != 1.0):
        raise RuntimeError(
            "Attempt to rescale model parameter which does not support rescaling"
        )
    return 1.0
