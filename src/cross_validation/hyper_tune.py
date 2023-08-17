from .cross_validate import cross_validate
from scipy.sparse import coo_array
import numpy as np
from MF_models import *


def hyper_tune(dataset: coo_array):
    R = dataset.toarray()

    dense_sgd = MF_GD()
    dense_batch = MF_GD()
    dense_mini_batch = MF_GD()
    dense_svd = MF_SVD()
    sparse_sgd = MF_GD()
    sparse_batch = MF_GD()
    sparse_mini_batch = MF_GD()
    sparse_svd = MF_SVD()

    models: list[tuple[MF_SVD | MF_GD, str]] = [
        (dense_sgd, "Dense SGD"),
        (dense_batch, "Dense Batch"),
        (dense_mini_batch, "Dense Mini Batch"),
        (dense_svd, "Dense SVD"),
        (sparse_sgd, "Sparse SGD"),
        (sparse_batch, "Sparse Batch"),
        (sparse_mini_batch, "Sparse Mini Batch"),
        (sparse_svd, "Sparse SVD"),
    ]

    n_factors_range = np.random.choice(np.arange(1, 30), 4, replace=False).tolist()
    lr_range = np.linspace(0.009, 0.07, num=4).tolist()
    epochs_range = np.random.choice(np.arange(20, 80), 4, replace=False).tolist()
    reg_range = np.linspace(0.0009, 0.009, num=4).tolist()
    batch_size_range = [2, 4, 8, 16, 32]

    for index, tup in enumerate(models):
        model, label = tup
        print(f"{index+1}/{len(models)} - Looking for best params for {label}")
        b_range = [] if not "Mini Batch" in label else batch_size_range
        cv_results = cross_validate(
            model=model,
            label=label,
            R=R,
            n_factors_range=n_factors_range,
            lr_range=lr_range,
            epochs_range=epochs_range,
            reg_range=reg_range,
            batch_size_range=b_range,
            n_folds=3,
        )

        if cv_results is None:
            print(
                f"All hyper-parameters combinations for {label} resulted in overflows"
            )
        else:
            best_params, mae, rmse = cv_results
            print(f"Best params for {label}: {best_params}, MAE: {mae}, RMSE: {rmse}")
