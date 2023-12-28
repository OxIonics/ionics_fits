import numpy as np

import ionics_fits as fits
from . import common


# Currently we don't fuzz this model. The tests are pretty comprehensive and
# should catch most issues. There are enough different heuristic code paths
# and it doesn't feel worth fuzzing them all!


def test_ms_time(plot_failures: bool):
    """Test for molmer_sorensen.MolmerSorensenTime"""

    def _test_ms_time(num_qubits, walsh_idx, start_excited, user_estimates):
        t = np.linspace(0, 1, 100) * 1e-6
        t_ref = 0.3e-6

        params = {
            "omega": np.array([0.5, 1, 2]) * np.pi / t_ref,
            "delta": 0,
            "n_bar": 0,
        }

        model = fits.models.MolmerSorensenTime(
            num_qubits=num_qubits, walsh_idx=walsh_idx, start_excited=start_excited
        )

        # There is a lot of covariance between delta and omega, so curvefit will
        # often fail to converge properly no matter how close the heuristic gets
        model.parameters["delta"].fixed_to = 0

        common.check_multiple_param_sets(
            t,
            model,
            params,
            common.TestConfig(
                plot_failures=plot_failures, param_tol=None, residual_tol=1e-4
            ),
            user_estimates=user_estimates,
        )

    _test_ms_time(num_qubits=1, walsh_idx=0, start_excited=True, user_estimates=[])
    for num_qubits in [1, 2]:
        for walsh_idx in [0, 1, 3]:
            # _test_ms_time(
            #     num_qubits=num_qubits,
            #     walsh_idx=walsh_idx,
            #     start_excited=False,
            #     user_estimates=[],
            # )
            _test_ms_time(
                num_qubits=num_qubits,
                walsh_idx=walsh_idx,
                start_excited=False,
                user_estimates=["omega"],
            )
            # _test_ms_time(
            #     num_qubits=num_qubits,
            #     walsh_idx=walsh_idx,
            #     start_excited=False,
            #     user_estimates=["delta"],
            # )
            # _test_ms_time(
            #     num_qubits=num_qubits,
            #     walsh_idx=walsh_idx,
            #     start_excited=False,
            #     user_estimates=["omega", "delta"],
            # )


def test_ms_freq(plot_failures: bool):
    """Test for molmer_sorensen.MolmerSorensenFreq"""

    def _test_ms_freq(num_qubits, walsh_idx, start_excited, user_estimates):
        w = np.linspace(-50e3, 50e3, 200) * 2 * np.pi
        t_pulse = 100e-6

        params = {
            "omega": np.array([0.5, 1, 2]) * np.pi / t_pulse,
            "w_0": np.array([0.25, 0, -0.125, 0.5]) * max(w),
            "n_bar": 0,
            "t_pulse": t_pulse,
        }

        model = fits.models.MolmerSorensenFreq(
            num_qubits=num_qubits, walsh_idx=walsh_idx, start_excited=start_excited
        )

        common.check_multiple_param_sets(
            w,
            model,
            params,
            common.TestConfig(
                plot_failures=plot_failures, param_tol=None, residual_tol=1e-4
            ),
            user_estimates=user_estimates,
        )

    # it's a bit excessive to run every permutation of these tests. If this gets
    # annoyingly slow we should pick a sensible subset to run
    _test_ms_freq(num_qubits=1, walsh_idx=0, start_excited=True, user_estimates=[])
    for num_qubits in [1, 2]:
        for walsh_idx in [0, 1, 3]:
            _test_ms_freq(
                num_qubits=num_qubits,
                walsh_idx=walsh_idx,
                start_excited=False,
                user_estimates=[],
            )
            _test_ms_freq(
                num_qubits=num_qubits,
                walsh_idx=walsh_idx,
                start_excited=False,
                user_estimates=["t_pulse", "omega"],
            )
            _test_ms_freq(
                num_qubits=num_qubits,
                walsh_idx=walsh_idx,
                start_excited=False,
                user_estimates=["t_pulse"],
            )
            _test_ms_freq(
                num_qubits=num_qubits,
                walsh_idx=walsh_idx,
                start_excited=False,
                user_estimates=["omega"],
            )
