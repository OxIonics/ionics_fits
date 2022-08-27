""" Randomized testing of fitting code. run with: poe fuzz """
from test_polynomial import TestPower, TestPolynomial

if __name__ == "__main__":
    classes = [TestPower, TestPolynomial]

    num_trials = 100  # TODO: could make this a command line arg if desired...
    for klass in classes:
        test = klass()
        test.setUp()
        test.fuzz(num_trials=num_trials)
