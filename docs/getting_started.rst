.. _getting_started:

Getting Started
===============

Installation
~~~~~~~~~~~~

Install from `pypi` with

.. code-block:: bash

   pip install ionics_fits

or add to your poetry project with

.. code-block:: bash

   poetry add ionics_fits

Basic usage
~~~~~~~~~~~

.. code-block:: python
   :linenos:

    import numpy as np
    from matplotlib import pyplot as plt

    from ionics_fits.models.polynomial import Line
    from ionics_fits.normal import NormalFitter

    a = 3.2
    y0 = -9

    x = np.linspace(-10, 10)
    y = a * x + y0

    fit = NormalFitter(x, y, model=Line())
    print(f"Fitted: y = {fit.values['a']:.3f} * x + {fit.values['y0']:.3f}")

    plt.plot(x, y)
    plt.plot(*fit.evaluate())
    plt.show()


This fits a test dataset to a line model. Here we've used a
:class:`~ionics_fits.normal.NormalFitter` which performs maximum-likelihood parameter
estimation, assuming normal statistics. This is the go-to fitter that's suitable in most
cases.

For more examples, see the :class:`~ionics_fits.common.Fitter` API documentation.
