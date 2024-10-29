import pytest
from GenZ.Astra_sim.get_astra_sim_time import divide_npus_count
import numpy as np

def test_divide_npus_count():
    # Test cases
    test_cases = [
        ({"npus_count":[2,4,4]}, [4,4,2], [[2, np.float64(2.0)], [np.float64(2.0), np.float64(2.0)], [np.float64(2.0)]], [[0, 1], [1, 2], [2]]),
        ({"npus_count": [2, 4, 4]}, [4, 4, 2], [[2, np.float64(2.0)], [np.float64(2.0), np.float64(2.0)], [np.float64(2.0)]], [[0, 1], [1, 2], [2]]),
        ({"npus_count": [8, 2, 3]}, [8, 2, 3], [[8], [2], [3]], [[0], [1], [2]]),
        ({"npus_count": [3, 12, 3]}, [9, 4, 3], [[3, np.float64(3.0)], [np.float64(4.0)], [3]], [[0, 1], [1], [2]]),
        ({"npus_count": [5, 2, 8]}, [10, 4, 2], [[5, 2], [4.0], [2.0]], [[0, 1], [2], [2]]),
        ({"npus_count": [6, 2, 2]}, [3, 2, 4], [[3.0], [2.0], [2, 2]],  [[0], [0], [1, 2]])
    ]

    # Run tests
    for config, sizes, golden_result, golden_dims in test_cases:
        result, dims = divide_npus_count(config, sizes)
        assert result == golden_result, f"Expected {golden_result}, but got {result}"
        assert dims == golden_dims, f"Expected {golden_dims}, but got {dims}"