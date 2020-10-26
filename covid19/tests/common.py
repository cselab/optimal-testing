from unittest import TestCase

import libepidemics
import numpy as np

def flatten(M: np.ndarray):
    """Flatten a numpy matrix."""
    return sum(M.tolist(), [])


def get_canton_model_data(K, days):
    """Return a random canton ModelData object.

    Arguments:
        K: number of cantons
        days: number of days (for commute data)
    """
    np.random.seed(12345)
    Mij = 1000 * np.random.rand(K, K)
    for i in range(K):
        Mij[i, i] = 0.0  # No self-flow.

    Cij = 1000 * np.random.rand(K, K)
    for i in range(K):
        Cij[i, i] = 0.0;

    return libepidemics.ModelData(
            region_keys=["C" + str(k) for k in range(K)],
            Ni=(1e6 + 1e6 * np.random.rand(K)).tolist(),
            Mij=flatten(Mij),
            Cij=flatten(Cij),
            ext_com_iu=flatten(100 * np.random.rand(days, K)),
            Ui=100 * np.random.rand(days))


class TestCaseEx(TestCase):
    def assertRelative(self, a, b, tolerance):
        if a == b:  # In the case both are 0.
            return
        relative = abs((a - b) / abs(a))
        if relative > tolerance:
            self.fail(f"assertRelative failed: |{a} - {b}| relative "
                      f"error {relative} larger than tolerance {tolerance}.")
