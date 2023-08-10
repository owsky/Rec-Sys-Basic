from itertools import product
from typing import List
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np
from MF_models import all_models_type, GD_models_type, SVD_models_type
from data.convert_dataset import convert_dataset


def cross_validate(
    model_cls: all_models_type,
    R: NDArray[np.float64],
    n_factors_range: List[int],
    lr_range: List[float],
    epochs_range: List[int],
    reg_range: List[float],
    n_folds=5,
    n_jobs=-1,
    batch_size_range: List[int] = [],
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    isBatched = "batch" in model_cls.__name__

    def perform_cross_validation(
        n_factors: int, lr: float, epochs: int, reg: float, batch_size: int = -1
    ):
        model = model_cls()
        isBatched = "batch" in model_cls.__name__
        total_mae = 0.0
        total_rmse = 0.0

        for train_idx, val_idx in kf.split(R):
            train_data = convert_dataset(R[train_idx], model_cls)
            val_data = R[val_idx]

            args = (
                (train_data, n_factors, lr, epochs, reg, batch_size)
                if isBatched
                else (train_data, n_factors, lr, epochs, reg)
            )

            model.fit(*args)  # type: ignore
            mae = model.accuracy_mae(val_data)
            total_mae += mae
            rmse = model.accuracy_rmse(val_data)
            total_rmse += rmse

        average_mae = total_mae / n_folds
        average_rmse = total_rmse / n_folds

        params = (
            (n_factors, epochs, lr, reg, batch_size)
            if isBatched
            else (n_factors, epochs, lr, reg)
        )

        return params, average_mae, average_rmse

    prod = (
        product(n_factors_range, lr_range, epochs_range, reg_range, batch_size_range)
        if isBatched
        else product(n_factors_range, lr_range, epochs_range, reg_range)
    )
    hyperparameter_combinations = [comb for comb in prod]
    with parallel_backend("loky", n_jobs=n_jobs):
        results: List[tuple[tuple[int, int, float, float], float, float]] = [
            result
            for result in Parallel(n_jobs=n_jobs)(
                delayed(perform_cross_validation)(*params)
                for params in tqdm(hyperparameter_combinations, desc="Processing jobs")
            )
            if result is not None
        ]
    best_params = min(results, key=lambda x: (x[1], x[2]))[0]
    return best_params
