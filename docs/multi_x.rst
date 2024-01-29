.. _multi_x:
Fitting to multi-dimensional data
=================================

`ionics-fits` is optimised for working with datasets with a single x-axis dimension,
however it does fully support datasets with multi-dimensional y-axis data.

:ref:`container` models can be used to extend the y-axis dimensionality of existing
models in various ways.

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

.. code-block:: python
   :linenos:

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
       x=(x_ax_0, x_ax_1), y=y.T, model=fits.multi_x.Gaussian2D()
   )
