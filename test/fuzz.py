"""Randomized testing of fitting code. run with: poe fuzz"""

import argparse
import logging
import time
import traceback

import matplotlib

from . import (
    common,
    test_benchmarking,
    test_exponential,
    test_gaussian,
    test_lorentzian,
    test_polynomial,
    test_rabi,
    test_rectangle,
    test_sinc,
    test_sinusoid,
    test_triangle,
)
from .common import Config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    targets = {
        "benchmarking": test_benchmarking.fuzz_benchmarking,
        "exponential": test_exponential.fuzz_exponential,
        "gaussian": test_gaussian.fuzz_gaussian,
        "lorentzian": test_lorentzian.fuzz_lorentzian,
        "polynomial": test_polynomial.fuzz_polynomial,
        "power": test_polynomial.fuzz_power,
        "rabi_freq": test_rabi.fuzz_rabi_freq,
        "rabi_freq_inverted": test_rabi.fuzz_rabi_freq_inverted,
        "rabi_time": test_rabi.fuzz_rabi_time,
        "rabi_time_inverted": test_rabi.fuzz_rabi_time_inverted,
        "rectangle": test_rectangle.fuzz_rectangle,
        "sinc": test_sinc.fuzz_sinc,
        "sinc2": test_sinc.fuzz_sinc2,
        "sinusoid": test_sinusoid.fuzz_sinusoid,
        "triangle": test_triangle.fuzz_triangle,
    }

    parser = argparse.ArgumentParser(
        description="Fit model fuzzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--continue-at-failure",
        action="store_true",
        help="If set, we do not stop fuzzing a target after a single failure",
    )
    parser.add_argument(
        "--plot-failures",
        action="store_true",
        help="If set, we plot data/fit for each failure",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of random parameter sets for each target",
    )
    parser.add_argument(
        "targets",
        nargs="?",
        type=str,
        help="models to fuzz, omit to fuzz all targets",
        choices=list(targets.keys()),
    )

    group = parser.add_argument_group("verbosity")
    group.add_argument(
        "-v", "--verbose", default=0, action="count", help="increase logging level"
    )
    group.add_argument(
        "-q", "--quiet", default=0, action="count", help="decrease logging level"
    )

    args = parser.parse_args()

    handler = logging.StreamHandler()
    logging.getLogger().setLevel(logging.WARNING + args.quiet * 10 - args.verbose * 10)
    logging.getLogger().addHandler(handler)

    args.targets = args.targets or targets.keys()
    args.targets = [args.targets] if isinstance(args.targets, str) else args.targets

    show = args.plot_failures
    if show:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Qt5Agg")
    else:
        matplotlib.use("agg")
        common.plt.show = lambda: None

    for target in args.targets:
        logger.info(f"Fuzzing {target}...")
        t0 = time.time()

        target_fun = targets[target]

        try:
            test_config = Config(plot_failures=args.plot_failures)
            failures = target_fun(
                num_trials=args.num_trials,
                stop_at_failure=not args.continue_at_failure,
                test_config=test_config,
            )
            if failures:
                logger.warning(
                    f"{failures} failure(s) out of {args.num_trials} "
                    f"({(1 - failures/args.num_trials) * 100:.1f} % success rate)"
                )
            else:
                t_trial = time.time() - t0
                logger.info(
                    f"success (took {t_trial:.1f} s for {args.num_trials} trails)"
                )
        except Exception:
            logger.warning(f"Failed with exception: {traceback.format_exc()}")
