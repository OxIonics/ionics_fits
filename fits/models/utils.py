import functools


def rename_params(param_map, unused_params):
    """Renames model parameters.

    :param model_class: the original fit model
    :param map: dictionary mapping the new names of parameters (dictionary keys) to the
        parameter names used by :param model_class: (dictionary values).
    :param unused_params: dictionary of parameters from :model_class: which are not used
        in the new class
    """

    def rename_impl(cls):
        original_params = [original_param for _, original_param in param_map.items()]

        if not set(original_params).union(set(unused_params)) == set(
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
            raise ValueError(
                "Unused parameters should not feature in the parameter map"
            )

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

        cls.func = func_wrapper(cls.func)

        def estimator_wrapper(func):
            @functools.wraps(func)
            def estimator_wrapper_impl(x, y, known_values, bounds):
                known_values = {
                    original_param: known_values[new_param]
                    for new_param, original_param in cls._PARAM_MAP.items()
                    if new_param in known_values.keys()
                }
                known_values.update(unused_params)

                param_guesses = func(x, y, known_values, bounds)
                param_guesses = {
                    new_param: param_guesses[original_param]
                    for new_param, original_param in cls._PARAM_MAP.items()
                }

                return param_guesses

            return estimator_wrapper_impl

        cls.estimate_parameters = estimator_wrapper(cls.estimate_parameters)

        return cls

    return rename_impl
