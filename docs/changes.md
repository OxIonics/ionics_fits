# Changelog

This document lists changes to `ionics_fits`.

Changes are divided into four categories:
* Bugfixes
* API changes - changes to the user-facing API. These will often be **breaking** (not backwards-compatible), requiring changes to code using `ionics_fits`.
* Other - small changes, such as tweaks to internals, which most users will not need to be aware of
* New models

## 1.3.0.5

### API changes
* **New feature**: Added `ReparametrizedModel` container
* **New feature**: Added `SineMinMax` model
* **New feature**: Added `Sine2` model
* **Breaking change**: `rescale_model_x` has been replaced with `ScaledModel` to make
  this feature consistent with the rest of the toolkit.

### API changes
* **New feature**: Added a sigmoid model

## 1.3.00

This release includes some invasive refactoring of `ionics_fits` internals to simplify
dataflow. This requires small changes to how models are defined (see breaking changes)
but will otherwise be transparent to users.

A major feature addition in this release is `ionics_fits` first foray into the world of multi-dimensional fitting.

### Bugfixes
* Incorrect calculation of pi times in the Rabi model

### API changes

* **Breaking**: `estimate_parameters` no longer takes a dictionary of model parameters as an input. Instead, it acts directly on the model's parameter dictionary.
* **Breaking**: all model parameters must now specify a scale function. This changes the previous behaviour where parameters with no scale function specified were treated as invariant.
* **Breaking**: scale functions are no longer passed the model as an input argument
* **New feature**: helper functions were introduced to `ionics_fits.common` to make specifying scale functions easier and more readable.
* **Breaking**: whether or not a model can be rescaled is now determined by the model's `can_rescale` method. All models must provide this method. Scale functions must now always return a float value. This replaces the previous behaviour where scale functions returning `None` was used to indicate a model that cannot be rescaled.
* **New feature**: enabling rescaling can now be set independently for the x and y axes
* **Breaking**: `Model`'s `get_initial_values` method has been removed
* **New feature**: `ModelParameter`s now raise exceptions if any fixed parameters have user estimates. This behaviour was previously allowed but was generally indicative of a bug
* **Breaking**: heuristics have now been moved into `ionics_fits.models.heuristics`. This includes functions, such as `param_min_srs`, which were previously methods of `Model`
* **Breaking**: removed a sampling heuristic which wasn't particularly useful
* **New feature**: solver arguments are now exposed via fitters
* **Breaking**: `AggregateModel` now takes a dictionary of models rather than a list of
  tuples.
* **Breaking**: `AggregateModel` now names parameters `{param_name}_{model_name}`
  instead of `{model_name}_{param_name}`

### New models
* `ConeSegment` model has been added

### Other changes
* Added a change log!
* Models now have `internal_parameters` as well as `parameters`. Internal parameters represent parameters which are not exposed directly to the user, but which still need to be stored and rescaled. They are used, for example, in the container models.
* Improved documentation
* The way rescaling is handled has been overhauled to improve dataflow. The `get_scaled_model` mechanism has been removed in favour of using `internal_parameters`. `ModelParameter`s and `Model`s now provide `rescale` and `unscale` methods to control rescaling. When a `ModelParameter` is rescaled, querying its attributes returns scaled values.
* `qt` is now a development dependency
* replaced `np.power` with the more pythonic (and faster) `**` operator where suitable
* ownership / copying has been tidied up and documented better
* `LaserRabi`: improved heuristic for estimating `eta` when `omega` is known
* Improved parameter estimator for `RepeatedModel`
* `RepeatedModel` now has the ability to aggregate derived parameter values, reporting
  only the values that are the same for all repetitions.


## Previous versions

Changes were not tracked prior to `v1.3`.