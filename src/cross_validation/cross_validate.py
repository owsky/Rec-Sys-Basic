from itertools import product
from typing import List
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np
from MF_models import all_models_type
from data.convert_dataset import convert_dataset


def cross_validate(
    model: all_models_type,
    label: str,
    R: NDArray[np.float64],
    n_factors_range,
    lr_range,
    epochs_range,
    reg_range,
    batch_size_range: List[int],
    n_folds=5,
):
    seed = 42
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    isBatch = "batch" in label.lower()
    isStochastic = "sgd" in label.lower()
    fixed_batch_size = -1 if isBatch else 1 if isStochastic else None

    def perform_cross_validation(
        n_factors: int, epochs: int, lr: float, reg: float, batch_size: int
    ):
        total_mae = 0.0
        total_rmse = 0.0

        params = (n_factors, epochs, lr, reg, batch_size)

        for train_idx, val_idx in kf.split(R):
            train_data = convert_dataset(R[train_idx], label)
            val_data = R[val_idx]

            with np.errstate(all="raise"):
                try:
                    model.fit(train_data, *params, seed=seed)
                except (RuntimeWarning, FloatingPointError) as rw:
                    return params, float("inf"), float("inf")

                mae = model.accuracy_mae(val_data)
                total_mae += mae
                rmse = model.accuracy_rmse(val_data)
                total_rmse += rmse

        average_mae = total_mae / n_folds
        average_rmse = total_rmse / n_folds

        return params, average_mae, average_rmse

    prod = (
        product(n_factors_range, epochs_range, lr_range, reg_range, [fixed_batch_size])
        if fixed_batch_size is not None
        else product(
            n_factors_range, epochs_range, lr_range, reg_range, batch_size_range
        )
    )
    hyperparameter_combinations = [comb for comb in prod]
    results: List[tuple[tuple[int, int, float, float], float, float]] = [
        result
        for result in Parallel(n_jobs=-1, backend="loky")(
            delayed(perform_cross_validation)(*params)
            for params in tqdm(hyperparameter_combinations, desc="Processing jobs")
        )
        if result is not None
    ]
    results_filtered = [param for param in results if param[1] != float("inf")]
    if len(results_filtered) > 0:
        return min(results_filtered, key=lambda x: (x[1], x[2]))
    else:
        return None
