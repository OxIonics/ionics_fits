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

Tools for writing parameter estimators.

.. automodule:: ionics_fits.models.heuristics
    :members:

Laser Rabi
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.laser_rabi
    :members:

Lorentzian
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.lorentzian
    :members:

Mølmer–Sørensen
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.molmer_sorensen
    :members:

Multi-X
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.multi_x
    :members:

Polynomial
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.polynomial
    :members:

Quantum Physics
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.quantum_phys
    :members:

Rabi
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.rabi
    :members:

Ramsey
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.ramsey
    :members:

Rectangle
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.rectangle
    :members:

Sigmoid
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sigmoid
    :members:

Sinc
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sinc
    :members:

Sinusoid
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.sinusoid
    :members:

Triangle
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.triangle
    :members:

Utils
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.utils
    :members:

Transformations
~~~~~~~~~~~~~~~
.. automodule:: ionics_fits.models.transformations

Aggregate Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.aggregate_model
    :members:

Mapped Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.mapped_model
    :members:

Model2D
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.model_2d
    :members:

Reparametrized Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.reparametrized_model
    :members:

Scaled Model
++++++++++++++++++++++++++++++++++++++++

.. automodule:: ionics_fits.models.transformations.scaled_model
    :members: