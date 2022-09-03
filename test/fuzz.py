""" Randomized testing of fitting code. run with: poe fuzz """
import argparse
import logging
import traceback

from test.test_polynomial import TestPower, TestPolynomial

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    targets = {"power": TestPower, "polynomial": TestPolynomial}

    parser = argparse.ArgumentParser(
        description="Fit model fuzzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--continue_at_failure",
        action="store_true",
        help="If set, we do not stop fuzzing a target after a single failure",
    )
    parser.add_argument(
        "--plot_failures",
        action="store_true",
        help="If set, we plot data/fit for each failure",
    )
    parser.add_argument(
        "--num_trials",
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

    for target in args.targets:
        logger.info(f"Fuzzing {target}...")

        test = targets[target]()
        test.setUp()

        try:
            failures = test.fuzz(
                num_trials=args.num_trials,
                stop_at_failure=not args.continue_at_failure,
                plot_failures=args.plot_failures,
            )
            if failures:
                logger.warning(
                    f"{failures} failure(s) out of {args.num_trials} "
                    f"({(1 - failures/args.num_trials) * 100:.1f} % success rate)"
                )
            else:
                logger.info("success")
        except Exception:
            logger.warning(f"Failed with exception: {traceback.format_exc()}")
