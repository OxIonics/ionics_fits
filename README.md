Lightweight Python library for data fitting with an emphasis on AMO (Atomic Molecular
and Optical physics) and Quantum Information.

`fits` was inspired by the [Oxford Ion Trap Group fitting library](https://github.com/OxfordIonTrapGroup/oitg/tree/master/oitg/fitting) originally
authored by @tballance. It is still in the alpha phase and is likely to be renamed
(although `fits` is still available on pypi)...please feel free to help bikeshed names.

# Getting started

## Installation

Install from `pypi` with `pip install ionics_fits` or add to your poetry project with
`poetry add fits`.

## Example Usage

Basic usage
```python
import numpy as np
import ionics_fits as fits

a = 3.2
y0 = -9

x = np.linspace(-10, 10)
y = a*x + y0

fit = fits.NormalFitter(x, y, model=fits.models.Line())
print(f"Fitted: y = {fit.values['a']:.3f} * x + {fit.values['y0']:.3f}")
```

The fit can be configured in various ways by modifying the model's `parameters`
dictionary (see the `fits.common.ModelParameter` class for more information). This
allows one to:
- change the bounds for parameters
- change which parameters are fixed to a constant value / floated
- supply initial values for parameters instead of relying on the heuristics

Example usage:
```python
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
plt.plot(*fit.evaluate(), '-.o', label="fit")
plt.grid()
plt.legend()
plt.show()
```

# Developing

Before committing:
- Update formatting: `poe fmt`
- Lints: `poe flake`
- Run test suite: `poe test`
- Optionally: [fuzz](#Fuzzing) any new models

# Design Philosophy

## Good Heuristics

Life is too short for failed fits. We can't guarantee to fit every dataset without any
help from the user (e.g. specifying initial parameter values) no matter how noisy or
incomplete it is...but we do our best!

Every fit model has a "parameter estimator" which uses heuristics to find good estimates
of the values of the model's free parameters. We expect these heuristics to be good
enough to allow the optimizer to fit any "reasonable" dataset. Fit failures are viewed
as a bug and we encourage our users to file issues where they find them (please post an
example dataset in the issue).

Currently this project is a MVP and many of the heuristics need some work. Expect there
to be cases where we could easily do better. Please report them where you find them!

## Validation

It's not enough to just fit the data, we want to know if we can trust the fit results
before acting on them. To assist validation, we provide a statistical estimate for the
fit quality. These tells the user how likely it is that their dataset could have arisen
through chance assuming the fitted model is correct given the assumed statistics.

There are two distinct aspects to the validation process: did the fit find the model
parameters which best match the data (as opposed to getting stuck in a local minimum in
parameter space)? and, are the fitted parameter values consistent with our prior
knowledge of the system (e.g. we know that a fringe contrast must lie within certain
bounds).

We encourage our users to approach both aspects of validation using the goodness of fit:
any prior knowledge about the system should be included in the fit setup through fixed
parameter values and parameter bounds; after that, a high fit significance indicates
both a good fit and that the system behaviour is consistent with our prior expectations.

## General purpose

This library is designed to be general purpose; rather than tackling specific problems
we try to target sets of problems -- we want to fit sinusoids not *your* sinusoid. This
is reflected, for example, in the choices of parametrisation, which are intended to be
extremely flexible, and the effort put into heuristics. If you find you can't easily fit
your sinusoid with the standard model/heuristics it's probably a bug in the model design
so please open an issue.

We encourage contributions of new fit models, but please consider generality before
submission. If you want to solve a specific problem in front of you, that's fine but
probably better suited to your own codebase.

## Extensibility

The library is designed to be extensible and ergonomic to user. Want to use different
statistics? Easy, just provide a new class that inherits from `FitBase`. Want to do some
custom post-fit processing? Override the `post_fit` method. Want to tweak the parameter
estimator for a model? Create a new model class that inherits from the original model
and modify away. If you're struggling to do what you want, it's probably a bug in the
library so report it.

At present the library is still an MVP. While we do provide some hooks like `post_fit`
there aren't many. The addition of further hooks will be driven by use cases, so please
open an issue if you find you can't easily extend the library in the way you want.

# Ontology

There are two main kinds of object in the library: `fit`s and `model`s. Models are
general-purpose functions to be fitted, such as sinusoids or Lorentzians, but are
agnostic about the statistics. Fits do the fitting (maximum likelihood parameter
estimation) and validation based on some underlying statistics (normal, binomial, etc). 

# Testing methodology

This package uses both `unit test`s and `fuzzing`.

## Unit Tests

- run using `poe test`
- to run a subset of tests use the `-k` flag e.g. `poe test -k "rabi"` to run only tests
  with the word `rabi` in their name. For more information about configuring pytest see
  the [documentation](https://docs.pytest.org/en/7.1.x/)
- all tests must pass before a PR can be merged into master
- PRs to add new models will only be merged once they have reasonable test coverage
- unit tests aim to provide good coverage over the space of "reasonable datasets". There
  will always be corner-cases where the fits fail and that's fine; the aim here is to
  cover the main cases users will hit in the wild
- when a user hits a case in the wild where the fit fails unexpectedly (i.e. we think
  the fit code should have handled it), a `regression test` based on the failing
  dataset should be added
- unit tests should be deterministic. Synthetic datasets should be included in the test
  rather than randomly generated at run time. Tip: while writing a test it's fine to let
  the test code generate datasets for you. Once you're happy, run the test in verbose
  mode and copy the dataset from the log output

## Fuzzing

- fuzzing is non-deterministic (random parameter values, randomly generated datasets)
  exploration of the parameter space.
- used when developing / debugging fits, but not automatically run by CI
- run with `poe fuzz` (see `--help` for details)
- fit failures during fuzzing are not automatically considered a bug; unlike unit tests,
  fuzzing explores the "unreasonable" part of parameter space as well. Indeed, a large
  part of the point of fuzzing is to help the developer understand what should be
  considered "reasonable" (this information should end up in the documentation for the
  fuzzed model).
