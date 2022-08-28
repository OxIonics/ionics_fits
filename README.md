Lightweight Python library for data fitting with an emphasis on AMO (Atomic Molecular and Optical physics) and Quantum Information.

It was inspired by the [Oxford Ion Trap Group fitting library](https://github.com/OxfordIonTrapGroup/oitg/tree/master/oitg/fitting) originally authored by @tballance. It is still in the alpha phase and is likely to be renamed (although `fits` is still available on pypi)...please feel free to help bikeshed names.

# Design Philosophy

## Good Heuristics

Life is too short for failed fits. We can't guarantee to fit every dataset without any help from the user (e.g. specifying initial parameter values) no matter how noisy or incomplete...but we do our best!

Every fit model has a "parameter estimator" which uses heuristics to find good estimates of the values of the model's free parameters. We expect these heuristics to be good enough to allow the optimizer to fit any "reasonable" dataset. Fit failures are viewed as a bug and we encourage our users to file issues where they find them (please post an example dataset in the issue).

Currently this project is a MVP and many of the heuristics need some work. Expect there to be cases where we could easily do better. Please report any issues (with the x and y data)!

## Validation

It's not enough to just fit the data, we want to know if we can trust the fit results before acting on them. To assist validation, we provide a statistical estimate for the fit quality. This tells the user how likely it is that their dataset could have arisen through chance assuming the fitted model is correct and specified statistics.

There are two distinct aspects to the validation process: did the fit find the model parameters which best match the data (as opposed to getting stuck in a local minimum in parameter space)? and, are the fitted parameter values consistent with our prior knowledge of the system (e.g. we know that a fringe contrast must lie within certain bounds).

We encourage our users to approach both aspects of validation using the goodness of fit: any prior knowledge about the system should be included in the fit setup through fixed parameter values and parameter bounds; after that, a high fit significance indicates both a good fit and that the system behaviour is consistent with our prior expectations.

## General purpose

This library is designed to be general purpose; rather than tackling specific problems we try to target sets of problems -- we to fit sinusoids not *your* sinusoid. This is reflected, for example, in the choices of parametrisation, which are intended to be extremely flexible, and the effort put into heuristics. So, if you find you can't fit your sinusoid with the model included here it's probably a bug in the model design so please open an issue.

We encourage contributions of new fit models, but please consider generality before submission. If you want to solve a specific problem in front of you, that's fine but probably better suited to your own codebase.

## Extensibility

The library is designed to be extensible and ergonomic to user. Want to use different statistics? Easy, just provide a new class that inherits from `FitBase`. Want to do some custom post-fit processing? Override the `post_fit` method. Want to tweak the parameter estimator for a model? Create a new model class that inherits from the original model and modify away. If you're struggling to do what you want, it's probably a bug in the library so report it.

At present the library is still an MVP. While we do provide some hooks like `post_fit` there aren't many. The addition of further hooks will be driven by use cases, so please open an issue if you find you can't easily extend the library in the way you want.

# Ontology

There are two main kinds of object in the library: `fit`s and `model`s. Models are general-purpose functions to be fitted, such as sinusoids or Lorentzians, but are agnostic about the statistics. Fits do the fitting (maximum likelihood parameter estimation) and validation based on some underlying statistics (normal, binomial, etc). 
