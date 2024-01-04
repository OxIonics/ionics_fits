.. _containers:

Helper utilities
================

Rescaling models
~~~~~~~~~~~~~~~~
Suppose you want to fit some frequency-domain Rabi oscillation data. However, the model
works in angular units, but your tooling needs linear units? No problem! Simply use the
:func:`ionics_fits.models.utils.rescale_model_x``` helper function.

.. code-block:: python

   detuning_model = fits.models.utils.rescale_model_x(fits.models.RabiFlopFreq, 2 * np.pi)

Or, suppose you actually want to scan the magnetic field and find the field offset which
puts the transition at a particular frequency?

.. code-block:: python

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
