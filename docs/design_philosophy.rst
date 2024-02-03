.. _design_philosophy:

Design Philosophy
=================

``ionics-fits`` is designed to be robust and flexible, while being "fast enough" - in
particular, data fitting should be much much faster than data acquisition.
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

``ionics-fits`` provides tools to help writing heuristics located in
:ref:`heuristics`.

General purpose
~~~~~~~~~~~~~~~

This library is designed to be general purpose; rather than tackling specific problems
we try to target sets of problems -- we want to fit sinusoids not some particular
dataset. This is reflected, for example, in the choices of parametrisation, which are
intended to be extremely flexible, and the effort put into heuristics. If you find you
can't easily fit your sinusoid with the standard model/heuristics it's probably a bug in
the model design so please open an issue.

We encourage contributions of new fit models, but please consider generality before
submission. If you want to solve a specific problem in front of you, that's fine but
probably better suited to your own codebase.

Dimensionality
~~~~~~~~~~~~~~

``ionics_fits`` supports fitting datasets with arbitrary x-axis and y-axis
dimensionality - so long as you have a fit model with the corresponding dimensionality.

:ref:`transformations` such as
:class:`~ionics_fits.models.transformations.repeated_model.RepeatedModel` and
:class:`~ionics_fits.models.transformations.model_2d.Model2D` provide a convenient
means of extending the dimensionality of existing models to make new models with
greater dimensionality.

Extensibility
~~~~~~~~~~~~~

The library is designed to be extensible. For example do you want
to:

* fit a dataset with different statistics? Easy, just provide a new class that inherits
  from :class:`ionics_fits.MLE.MLEFitter`.
* do some custom post-fit processing? Override the
  :meth:`ionics_fits.common.Model.calculate_derived_params` method.
* Want to tweak the parameter estimator for a model? Create a new model class that
  inherits from the original model and modify away.
* Fit a dataset in Hertz, while the model uses radians / s? Use the
  :class:`~ionics_fits.models.transformations.scaled_model.ScaledModel` transformation.
* Change how a model is parametrised? Use the
  :class:`~ionics_fits.models.transformations.reparametrized_model.ReparametrizedModel`
  transformation.

If you're struggling to do what you want, it's probably a bug in the library so please
report it.

Testing
~~~~~~~

``ionics_fits`` is extensively tested. See :ref:`testing` for further details.
