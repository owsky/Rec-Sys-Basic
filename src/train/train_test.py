from MF_models import MF_base, R_type
from numpy.typing import NDArray
import numpy as np
from .default_params import DefaultParams


def train_test(
    model: MF_base,
    hyper_params: DefaultParams,
    trainset: R_type,
    testset: NDArray[np.float64],
    seed: int | None = None,
):
    if seed != None:
        model.fit(trainset, *hyper_params, seed=seed)
    else:
        model.fit(trainset, *hyper_params)
    return model.accuracy_mae(testset), model.accuracy_rmse(testset)
