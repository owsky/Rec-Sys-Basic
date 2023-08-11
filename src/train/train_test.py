from MF_models import MF_base
from numpy.typing import NDArray
import numpy as np
from scipy.sparse import dok_array, coo_matrix, csr_matrix


def train_test(
    model: MF_base,
    hyper_params,
    trainset: NDArray[np.float64] | coo_matrix | dok_array | csr_matrix,
    testset: NDArray[np.float64],
    seed: int | None = None,
):
    if seed != None:
        model.fit(trainset, *hyper_params, seed=seed)
    else:
        model.fit(trainset, *hyper_params)
    return model.accuracy_mae(testset), model.accuracy_rmse(testset)
