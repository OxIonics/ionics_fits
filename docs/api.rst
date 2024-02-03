.. _api:

ionics-fits API
=================

Common
~~~~~~
.. automodule:: ionics_fits.common
    :members:
    :special-members: __call__, __init__
    :private-members: _func, _fit

Utils
~~~~~

.. automodule:: ionics_fits.utils
    :members:

.. _fitters:

Fitters
~~~~~~~

.. automodule:: ionics_fits.normal
    :members:
    :special-members: __init__
.. automodule:: ionics_fits.MLE
    :members:
    :special-members: __init__
.. automodule:: ionics_fits.binomial
    :members:
    :special-members: __init__

Validators
~~~~~~~~~~
.. automodule:: ionics_fits.validators
    :members:
    :special-members: __init__


Models
~~~~~~~~~~~~
.. automodule:: ionics_fits.models
    :members:

Benchmarking
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.benchmarking
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Cone
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.cone
    :members:
    :private-members: _func
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Exponential
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.exponential
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Gaussian
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.gaussian
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale
    
.. _heuristics:

Heuristics
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.heuristics
    :members:

Laser Rabi
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.laser_rabi
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale
    

Lorentzian
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.lorentzian
    :members:
    :private-members: _func
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale
    

Mølmer–Sørensen
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.molmer_sorensen
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Multi-X
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.multi_x
    :members:

Polynomial
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.polynomial
    :members:
    :exclude-members: Power

.. autoclass:: ionics_fits.models.polynomial.Power
    :members:
    :special-members: __init__
    :private-members: _func
    :exclude-members: estimate_parameters, optimal_n, rescale, func, get_num_x_axes, get_num_y_axes, can_rescale


Quantum Physics
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.quantum_phys
    :members:

Rabi
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.rabi
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Ramsey
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.ramsey
    :members:
    :private-members: _func
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Rectangle
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.rectangle
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Sigmoid
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sigmoid
    :members:
    :private-members: _func
    :special-members: __init__
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Sinc
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sinc
    :members:
    :private-members: _func
    :exclude-members: func, estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale

Sinusoid
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sinusoid
    :members:
    :private-members: _func
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, bound_param_values, bound_param_uncertainties, new_param_values

Triangle
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.triangle
    :members:
    :private-members: _func
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale


Utils
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.utils
    :members:
    :exclude-members: clip

Transformations
~~~~~~~~~~~~~~~
.. automodule:: ionics_fits.models.transformations

Aggregate Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.aggregate_model
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func

Mapped Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.mapped_model
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func

Model2D
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.model_2d
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func

Reparametrized Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.reparametrized_model
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func

Repeated Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.repeated_model
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func

Scaled Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.scaled_model
    :members:
    :special-members: __init__
    :exclude-members: estimate_parameters, get_num_x_axes, get_num_y_axes, can_rescale, calculate_derived_params, func
