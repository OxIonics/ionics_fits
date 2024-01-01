Lightweight Python library for data fitting with an emphasis on AMO (Atomic Molecular
and Optical physics) and Quantum Information.

`ionics_fits` was inspired by the [Oxford Ion Trap Group fitting library](https://github.com/OxfordIonTrapGroup/oitg/tree/master/oitg/fitting) originally
authored by @tballance.

# Getting started

## Installation

Install from `pypi` with `pip install ionics_fits` or add to your poetry project with
`poetry add ionics_fits`.

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
- supply initial estimates parameters instead of relying on the heuristics

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
plt.plot(*fit.evaluate(True), '-.o', label="fit")
plt.grid()
plt.legend()
plt.show()
```

Models provide a callable interface (see `Model.__call__` for details) to make using
them outside of fits convenient. For example:

```python
import numpy as np
import ionics_fits as fits

x = np.linspace(0, 1, 100) * 2 * np.pi
sinusoid = fits.models.Sinusoid()
y = sinusoid(x=x, a=1, omega=2, phi=0, y0=0)
```

## Multi-dimensional datasets

`ionics-fits` is optimised for working with datasets with a single x-axis dimension,
however it does fully support datasets with multi-dimensional y-axis data. Each y-axis
is referred to as a "y channel".

[Container](##Containers) models can be used to extend the y-axis dimensionality of
existing models in various ways.

Working with models with higher x-axis dimensionality presents a number of interesting
challenges - particularly around how one writes general-purpose, robust heuristics -
which are out of scope for this package.

It does, however, have some limited support for data sets with multiple x-axis
dimensions through [`ionics_fits.multi_x`](../master/ionics_fits/multi_x/common.py).
This allows one to fit datasets with multiple x axes hierarchically, by sequentially
fitting one model to each x-axis with the results from one fit being the input to the
next fit - see :class ionics_fits.multi_x.Model2D: for more details.

Examples of fitting datasets with multiple x-axes (see the tests in
[`test.multi_x`](../master/test/multi_x.py)) for more examples):

```python
# Fit a 2D Gaussian
params = {
    "a": 5,
    "x0_x": -2,
    "x0_y": +0.5,
    "sigma_x": 2,
    "sigma_y": 5,
    "y0": 1.5,
}

def gaussian(x, y, a, x0_x, x0_y, sigma_x, sigma_y, y0):

    A = a / (sigma_x * np.sqrt(2 * np.pi)) / (sigma_y * np.sqrt(2 * np.pi))

    return (
        A
        * np.exp(
            -(((x - x0_x) / (np.sqrt(2) * sigma_x)) ** 2)
            - (((y - x0_y) / (np.sqrt(2) * sigma_y)) ** 2)
        )
        + y0
    )


x_mesh_0, x_mesh_1 = np.meshgrid(x_ax_0, x_ax_1)

y = gaussian(x_mesh_0, x_mesh_1, **params)
model = fits.multi_x.Gaussian2D()
fit = fits.multi_x.common.Fitter2D(
    x=[x_ax_0, x_ax_1], y=y.T, model=fits.multi_x.Gaussian2D()
)
```

```python
# Simulate Rabi flopping on a blue sideband as a function of the alignment between the
# laser and the motional mode

t_pi = 5e-6
omega = np.pi / t_pi
eta = 0.1
theta_0 = 0.25

angle_axis = np.linspace(-np.pi / 2, +np.pi / 3, 50)
time_axis = np.linspace(0, 5 * (t_pi / eta), 75)
time_mesh, angle_mesh = np.meshgrid(time_axis, angle_axis)

flop_model = fits.models.LaserFlopTimeThermal(
    start_excited=False, sideband_index=+1, n_max=1
)
flop_model.parameters["n_bar"].fixed_to = 0
flop_model.parameters["delta"].fixed_to = 0
flop_model.parameters["omega"].fixed_to = omega
flop_model.parameters["P_readout_e"].fixed_to = 1
flop_model.parameters["P_readout_g"].fixed_to = 0

sinusoid_model = fits.models.Sinusoid()
sinusoid_model.parameters["omega"].fixed_to = 1
sinusoid_model.parameters["x0"].fixed_to = -np.pi / 2
sinusoid_model.parameters["y0"].fixed_to = 0
sinusoid_model.parameters["phi"].offset = 0

# Generate data to fit
y = np.zeros_like(time_mesh)
for idx, angle in np.ndenumerate(angle_axis):
    eta_angle = sinusoid_model(x=angle, a=eta, phi=theta_0)
    y[idx, :] = flop_model(x=time_axis, eta=eta_angle, omega=omega)

model = fits.multi_x.Model2D(
    models=[flop_model, sinusoid_model],
    result_params=["eta"],
)

params = {"omega_x0": omega, "x0_x1": -np.pi / 2}
fit = fits.multi_x.common.Fitter2D(x=[time_axis, angle_axis], y=y.T, model=model)
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

Helper tools for writing heuristics are located in [`models.heuristics`](../master/ionics_fits/models/heuristics.py).

## Validation

It's not enough to just fit the data, we want to know if we can trust the fit results
before acting on them.  There are two distinct aspects to the validation problem: did
the fit find the model parameters which best match the data (as opposed to getting stuck
in a local minimum in parameter space far from the global optimum)? and, are the fitted
parameter values consistent with our prior knowledge of the system (e.g. we know that a
fringe contrast must lie within certain bounds).

First, any prior knowledge about the system should be incorporated by specifying fixed
parameter values and parameter bounds. After that, the fit is validated using a
[`FitValidator`](../master/ionics_fits/validators.py). Validators provide a flexible
and extensible framework for using statistical tests to validate fits.

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
statistics? Easy, just provide a new class that inherits from `MLEFitter`. Want to do some
custom post-fit processing? Override the `calculate_derived_parameters` method. Want to
tweak the parameter estimator for a model? Create a new model class that inherits from
the original model and modify away. If you're struggling to do what you want, it's
probably a bug in the library so please report it.

`ionics_fits` provides a number of tools to make it easier to extend models. See, for
example [`models.utils`](../master/ionics_fits/models/utils.py) and [`models.containers`](../master/ionics_fits/models/containers.py). Suppose you want to...
## Rescaling models

...fit some frequency-domain Rabi oscillation data. However, the model works in angular
units, but your tooling needs linear units. No problem! Simply use the `rescale_model_x`
tool.

```python
detuning_model = fits.models.utils.rescale_model_x(fits.models.RabiFlopFreq, 2 * np.pi)
```

Or, suppose you actually want to scan the magnetic field and find the field offset which
puts the transition at a particular frequency?
```python
class _RabiBField(fits.models.RabiFlopFreq):
    def __init__(self, dfdB, B_0, f_0, start_excited):
        super().__init__(start_excited=start_excited)
        self.dfdB = dfdB
        self.B_0 = B_0
        self.f_0 = f_0

    def calculate_derived_params(self, x, y, fitted_params, fit_uncertainties):
        derived_params, derived_uncertainties = super().calculate_derived_params(
            x, y, fitted_params, fit_uncertainties
        )

        df = derived_params["f_0"] - self.f_0
        dB = df / self.dfdB
        B_0 = dB + self.B_0

        derived_params["B_0"] = B_0
        derived_uncertainties["B_0"] = derived_uncertainties["f_0"] * self.dfdB

        return derived_params, derived_uncertainties

RabiBField = fits.models.utils.rescale_model_x(_RabiBField, 2 * np.pi * dfdB)
```

## Containers

...fit multiple independent models simultaneously and do some post-processing on the
results. Use an `AggregateModel`.

```python
class LineAndTriange(fits.models.AggregateModel):
  def __init__(self):
    line = fits.models.Line()
    triangle = fits.models.Triangle()
    super().__init__(models={"line": line, "triangle": triangle})

  def calculate_derived_params(
      self,
      x: Array[("num_samples",), np.float64],
      y: Array[("num_samples",), np.float64],
      fitted_params: Dict[str, float],
      fit_uncertainties: Dict[str, float],
  ) -> Tuple[Dict[str, float], Dict[str, float]]:
      derived_params, derived_uncertainties = super().calculate_derived_params()
      # derive new results from the two fits
      return derived_params, derived_uncertainties
```

`AggregateModel`s also support joint fitting of datasets via "common" parameters, whose value is the same for all. For example

```python
# Make a model that fits time scans of red and blue sidebands simultaneously
# All parameters are fit jointly
rsb = fits.models.LaserFlopTimeThermal(start_excited=False, sideband_index=-1)
bsb = fits.models.LaserFlopTimeThermal(start_excited=False, sideband_index=+1)

model = fits.models.AggregateModel(
    models={"rsb": rsb, "bsb": bsb},
    common_params={
        param: (rsb.parameters[param], [("rsb", param), ("bsb", param)])
        for param in rsb.parameters.keys()
    },
)
```

...use the single-qubit Rabi flop model to fit simultaneous Rabi flopping on multiple
qubits at once with some parameters shared and some independent.

```python
class MultiRabiFreq(fits.models.RepeatedModel):
    def __init__(self, n_qubits):
        super().__init__(
            model=fits.models.RabiFlopFreq(start_excited=True),
            common_params=[
                "P_readout_e",
                "P_readout_g",
                "t_pulse",
                "omega",
                "tau",
                "t_dead",
            ],
            num_repetitions=n_qubits,
        )

```
## And more!

At present the library is still an MVP. Further work will be driven by use cases, so
please open an issue if you find you can't easily extend the library in the way you
want.

# Ontology

There are three main kinds of object in the library: `ModelParameter`s, `Model`s
and `Fitters`. `ModelParameter`s represent the parameters for a given model. They
are used to store metadata, such as fit bounds. `Model`s are general-purpose
functions to be fitted, such as sinusoids or Lorentzians, but are
agnostic about the statistics. `Fitter`s do the fitting (maximum likelihood parameter
estimation).

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
