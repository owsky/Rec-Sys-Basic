import numpy as np
from numpy.typing import NDArray


def sparsity(matrix: NDArray[np.number]) -> float:
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    zero_elements = total_elements - non_zero_elements
    return zero_elements / total_elements
