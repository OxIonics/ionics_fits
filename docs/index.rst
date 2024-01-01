Ionics Fits
===========

Lightweight python fitting library with an emphasis on Atomic Molecular and Optical
Physics.

`ionics_fits` was inspired by the `Oxford Ion Trap Group fitting library <https://github.com/OxfordIonTrapGroup/oitg/tree/master/oitg/fitting>`_ originally authored by 
`@tballance <https://github.com/tballance>`_.

.. contents::
   :depth: 3


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

Basic Usage
~~~~~~~~~~~

There are three main types of object in `ionics_fits`: :class:`ModelParameter`,
:class:`Model` and :class:`Fitter`.

A :class:`Model` represents a fit function and the other information we need to fit it
robustly, such as a dictionary of its parameters and heuristics used to find a good
starting point for the fitter. :class:`Model` are agnostic about the statistics of the
datasets they are used to fit.

A :class:`ModelParameter` represents a parameter of the model, along with metadata such
as the limits the parameter is allowed to vary over.

:class:`Fitter` perform maximum-likelihood parameter estimation to fit datasets to
models. A number of fitters are provided to handle common statistical distributions,
such as normal and binomial distrubutions.

Let's start with a simple example to see how this works:

.. code-block:: python
   :linenos:

   import numpy as np
   import ionics_fits as fits

   a = 3.2
   y0 = -9

   x = np.linspace(-10, 10)
   y = a*x + y0

   fit = fits.NormalFitter(x, y, model=fits.models.Line())
   print(f"Fitted: y = {fit.values['a']:.3f} * x + {fit.values['y0']:.3f}")

This fits a test dataset to a line model. Here we've used a :class:`NormalFitter`
which performs maximum-likelihood parameter estimation, assuming normal statistics.
This is the go-to fitter that's suitable in most cases.

Customising the fit
~~~~~~~~~~~~~~~~~~~

The fit can be configured in various ways by modifying the model's parameter
dictionary. This
allows one to:
- change the bounds for parameters
- change which parameters are fixed to a constant value / floated
- supply initial estimates parameters instead of relying on the heuristics

Example usage:

.. code-block:: python
   :linenos:

   import numpy as np
   from matplotlib import pyplot as plt
   import ionics_fits as fits

   # Example problem: fit the amplitude and phase of a sinusoid whose frequency is known
   # exactly
   omega = 2 * np.pi  # we know the frequency
   model = fits.models.Sinusoid()
   model.parameters["omega"].fixed_to = omega

   # generate synthetic data to fit
   params = {
       "a": np.random.uniform(low=1, high=5),
       "omega": omega,
       "phi": np.random.uniform(-1, 1) * 2 * np.pi,
       "y0": 0,
       "x0": 0,
       "tau": np.inf,
   }
   x = np.linspace(-3, 3, 100)
   y = model.func(x, params)

   fit = fits.NormalFitter(x, y, model=model)
   print(f"Amplitude: dataset = {params['a']:.3f}, fit = {fit.values['a']:.3f}")
   print(f"Phase: dataset = {params['phi']:.3f}, fit = {fit.values['phi']:.3f}")

   plt.plot(x, y, label="data")
   plt.plot(*fit.evaluate(True), '-.o', label="fit")
   plt.grid()
   plt.legend()
   plt.show()


Using models outside of a fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models provide a callable interface to make using them outside of fits convenient. For
example:

.. code-block:: python
   :linenos:

   import numpy as np
   import ionics_fits as fits

   x = np.linspace(0, 1, 100) * 2 * np.pi
   sinusoid = fits.models.Sinusoid()
   y = sinusoid(x=x, a=1, omega=2, phi=0, y0=0)

Design Philosophy
=================

`ionics-fits` is designed to be extremely robust and flexible, while being "fast
enough" - in particular, data fitting should be much much faster than data acquisition.
However, optimising for speed is explicitly not a design goal - for example, we
generally prefer heuristics which are robust over faster ones which have less good
coverage.

Heuristics
~~~~~~~~~~

Life is too short for failed fits. We can't guarantee to fit every dataset without any
help from the user (e.g. specifying initial parameter values) no matter how noisy or
incomplete it is...but we do our best!

Every fit model has a "parameter estimator" which uses heuristics to find good estimates
of the values of the model's free parameters. We expect these heuristics to be good
enough to allow the optimizer to fit any "reasonable" dataset. Fit failures are viewed
as a bug and we encourage our users to file issues where they find them (please post an
example dataset in the issue).

`ionics-fits` provides tools to help writing heuristics located in
:ref:`heuristics`.

General purpose
~~~~~~~~~~~~~~~

This library is designed to be general purpose; rather than tackling specific problems
we try to target sets of problems -- we want to fit sinusoids not *your* sinusoid. This
is reflected, for example, in the choices of parametrisation, which are intended to be
extremely flexible, and the effort put into heuristics. If you find you can't easily fit
your sinusoid with the standard model/heuristics it's probably a bug in the model design
so please open an issue.

We encourage contributions of new fit models, but please consider generality before
submission. If you want to solve a specific problem in front of you, that's fine but
probably better suited to your own codebase.

Extensibility
~~~~~~~~~~~~~

The library is designed to be extensible and ergonomic to user. For example do you:
* Want to use different statistics? Easy, just provide a new class that inherits from
:class:`MLEFitter`.
* Want to do some custom post-fit processing? Override the
:meth:`calculate_derived_parameters` method.
* Want to tweak the parameter estimator for a model? Create a new model class that
inherits from the original model and modify away.
* Fit a dataset in Hertz, while the model uses radians / s? Use the
:func:`ionics_fits.models.utils.rescale_model_x` helper.

If you're struggling to do what you want, it's probably a bug in the library so please
report it.

Contributing
============


Before committing:

* Check formatting: ``poe fmt``
* Lints: ``poe flake``
* Run test suite: ``poe test``
* Optionally, any new models: ``poe fuzz``

Testing methodology
===================

`ionics-fits` is heavily tested using both unit tests and fuzzing.

Unit Tests
~~~~~~~~~~

* run using ``poe test``
* to run a subset of tests use the ``-k`` flag e.g. ``poe test -k "rabi"`` to run onlytests
  with the word `rabi` in their name. For more information about configuring pytest see
  the `documentation <https://docs.pytest.org/en/7.1.x/>`_
* all tests must pass before a PR can be merged into master
* PRs to add new models will only be merged once they have reasonable test coverage
* unit tests aim to provide good coverage over the space of "reasonable datasets". There
  will always be corner-cases where the fits fail and that's fine; the aim here is to
  cover the main cases users will hit in the wild
* when a user hits a case in the wild where the fit fails unexpectedly (i.e. we think
  the fit code should have handled it), a `regression test` based on the failing
  dataset should be added
* unit tests should be deterministic. Synthetic datasets should be included in the test
  rather than randomly generated at run time. Tip: while writing a test it's fine to let
  the test code generate datasets for you. Once you're happy, run the test in verbose
  mode and copy the dataset from the log output

Fuzzing
~~~~~~~~~~

* fuzzing is non-deterministic (random parameter values, randomly generated datasets)
  exploration of the parameter space.
* used when developing / debugging fits, but not automatically run by CI
* run with ``poe fuzz`` (see ``--help`` for details)
* fit failures during fuzzing are not automatically considered a bug; unlike unit tests,
  fuzzing explores the "unreasonable" part of parameter space as well. Indeed, a large
  part of the point of fuzzing is to help the developer understand what should be
  considered "reasonable" (this information should end up in the documentation for the
  fuzzed model).
* fuzzing is considered a tool to help developers finding corner-cases. We don't aim
  for the same level of code coverage with fuzzing that we do with the unit tests which
  *should*, for example, cover ever code path in every parameter estimator


Contents
========

.. toctree::
   :maxdepth: 2

   utils.rst
   containers.rst
   multi_x.rst
   validators.rst
   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`