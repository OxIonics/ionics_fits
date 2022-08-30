import functools
from typing import Callable, Dict

from ..common import FitModel


def rename_params(
    param_map: Dict[str, str], unused_params: Dict[str, float]
) -> Callable[[FitModel], FitModel]:
    """Rename rename model parameters.

    :param map: dictionary mapping names of parameters in the new model to names of
        parameters used in the original model.
    :param unused_params: dictionary mapping names of parameters used in the original
        model to values they are fixed to in the new model. These will not be parameters
        of the new model.
    :returns: function which takes in the original model class to be wrapped and returns
        the new (wrapped) model.
    """

    def rename_impl(cls):
        original_params = [original_param for _, original_param in param_map.items()]

        if not set(original_params).union(set(unused_params.keys())) == set(
            cls._PARAMETERS.keys()
        ):
            raise ValueError(
                "Parameter map does not match original model class parameters"
            )

        if not set(unused_params.keys()).issubset(set(cls._PARAMETERS.keys())):
            raise ValueError(
                "Unused parameters must be parameters of original model class"
            )

        if set(unused_params.keys()).intersection(set(original_params)) != set():
            raise ValueError("Unused parameters must not feature in the parameter map")

        cls._PARAMETERS = {
            new_param: cls._PARAMETERS[original_param]
            for new_param, original_param in param_map.items()
        }
        cls._PARAM_MAP = dict(param_map)
        cls._UNUSED_PARAMS = dict(unused_params)

        def func_wrapper(func):
            @functools.wraps(func)
            def func_wrapper_impl(x, params):
                params = {
                    original_param: params[new_param]
                    for new_param, original_param in cls._PARAM_MAP.items()
                }
                params.update(cls._UNUSED_PARAMS)
                return func(x, params)

            return func_wrapper_impl

        def estimator_wrapper(estimator):
            @functools.wraps(estimator)
            def estimator_wrapper_impl(x, y, known_values, bounds):
                known_values = {
                    original_param: known_values[new_param]
                    for new_param, original_param in cls._PARAM_MAP.items()
                    if new_param in known_values.keys()
                }
                known_values.update(unused_params)

                bounds = {
                    cls._PARAM_MAP[new_param]: bounds
                    for new_param, bounds in bounds.items()
                }
                bounds.update(
                    {
                        original_param: (value, value)
                        for original_param, value in cls._UNUSED_PARAMS.items()
                    }
                )

                param_guesses = estimator(x, y, known_values, bounds)
                param_guesses = {
                    new_param: param_guesses[original_param]
                    for new_param, original_param in cls._PARAM_MAP.items()
                }

                return param_guesses

            return estimator_wrapper_impl

        cls.func = func_wrapper(cls.func)
        cls.estimate_parameters = estimator_wrapper(cls.estimate_parameters)

        return cls

    return rename_impl
