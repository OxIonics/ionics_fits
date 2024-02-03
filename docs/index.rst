Ionics Fits
===========

Python data fitting library with an emphasis on Atomic Molecular and Optical Physics.

``ionics_fits`` was inspired by the `Oxford Ion Trap Group fitting library <https://github.com/OxfordIonTrapGroup/oitg/tree/master/oitg/fitting>`_ originally authored by 
`@tballance <https://github.com/tballance>`_.

Overview
========

There are four main types of object in ``ionics_fits``:

* :class:`~ionics_fits.common.ModelParameter`
* :class:`~ionics_fits.common.Model`
* :class:`~ionics_fits.common.Fitter`
* :class:`~ionics_fits.validators`

The main source of documentation is the :ref:`api` manual. This has documentation for
all classes in ``ionics_fits``\, with worked examples.

``ionics_fits`` aims to provide extensive, user-friendly documentation. If you can't
easily find the information you need to use it, we consider that a bug so please
open an issue.


Model Parameters
~~~~~~~~~~~~~~~~

A :class:`~ionics_fits.common.ModelParameter` represents a parameter of the model, along
with metadata such as the limits the parameter is allowed to vary over, wheter it is
fixed to a constant value or floated during the fit, etc.

Models
~~~~~~

A :class:`~ionics_fits.common.Model` represents a fit function along with a dictionary of
its parameters and heuristics used to find a good starting point for the fitter.
:class:`~ionics_fits.common.Model`\s are agnostic about the statistics of the datasets
they are used to fit.

:class:`~ionics_fits.common.Model` provides a callable interface to make it covenient
to use them outside of fits. See :func:`ionics_fits.common.Model.__call__` for details.

:ref:`transformations` are a special class of models which modify the behaviour of other
models, for example allowing a model to be reparametrized or combining existing models
to create a new model with greater dimensionality.

Fitters
~~~~~~~

:class:`~ionics_fits.common.Fitter`\s perform maximum-likelihood parameter estimation to
fit datasets to models. A number of :ref:`fitters` are provided to handle common
statistical distributions, such as Normal and Binomial distributions.

Validators
~~~~~~~~~~

It's not enough to just fit the data, we want to know if we can trust the fit results
before acting on them. This is the job of
:class:`~ionics_fits.validators.FitValidator`\s. See the :ref:`validators` API
documentation for details.


Contents
========


.. toctree::
   :maxdepth: 2

   getting_started.rst
   design_philosophy.rst
   contributing.rst
   testing.rst
   api.rst
   changes.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`