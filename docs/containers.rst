.. _containers:
Container models
=================

Suppose you want to fit multiple independent models simultaneously and do some
post-processing on the results? Use an :class:`ionics_fits.models.containers.AggregateModel`.

.. code-block:: python
   :linenos:

   class LineAndTriange(fits.models.AggregateModel):
     def __init__(self):
       line = fits.models.Line()
       triangle = fits.models.Triangle()
       super().__init__(models=[("line", line), "triangle", triangle])

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


Suppose you wnat to use the single-qubit Rabi flop model to fit simultaneous Rabi
flopping on multiple qubits at once with some parameters shared and some independent.

.. code-block:: python
   :linenos:

   class MultiRabiFreq(fits.models.RepeatedModel):
       def __init__(self, n_qubits):
           super().__init__(
               inner=fits.models.RabiFlopFreq(start_excited=True),
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
