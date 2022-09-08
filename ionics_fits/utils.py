import numpy as np
from typing import List, Union


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
