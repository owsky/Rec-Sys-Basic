from tabulate import tabulate
from data.convert_dataset import convert_dataset
from .train_test import train_test
from scipy.sparse import coo_array
from lightfm.cross_validation import random_train_test_split
from utils import runtime
from tqdm import tqdm
from MF_models import *
from .default_params import *


def aux(model, hyper_params, label: str, trainset, testset):
    trainset = convert_dataset(trainset, label)
    return label, *runtime(
        lambda: train_test(
            model, hyper_params, trainset=trainset, testset=testset, seed=42
        )
    )


def runtimes_all(dataset: coo_array):
    trainset, testset = random_train_test_split(
        dataset, test_percentage=0.2, random_state=42
    )
    testset = testset.toarray()

    dense_sgd = MF_GD_all()
    dense_batch = MF_GD_all()
    dense_mini_batch = MF_GD_all()
    dense_svd = MF_SVD_all()
    sparse_sgd = MF_GD_all()
    sparse_batch = MF_GD_all()
    sparse_mini_batch = MF_GD_all()
    sparse_svd = MF_SVD_all()

    models = [
        # (dense_sgd, DenseStochasticParams, "Dense SGD"),
        # (dense_batch, DenseBatchParams, "Dense Batch"),
        # (dense_mini_batch, DenseMiniBatchParams, "Dense Mini Batch"),
        (dense_svd, DenseSVDParams, "Dense SVD"),
        # (sparse_sgd, SparseStochasticParams, "Sparse SGD"),
        # (sparse_batch, SparseBatchParams, "Sparse Batch"),
        # (sparse_mini_batch, SparseMiniBatchParams, "Sparse Mini Batch"),
        # (sparse_svd, SparseSVDParams, "Sparse SVD"),
    ]

    results = map(
        lambda tup: aux(tup[0], tup[1], tup[2], trainset, testset),
        tqdm(iterable=models, desc="Computing training runtimes"),
    )

    table = []
    for label, runtime, acc in results:
        mae, rmse = acc
        table.append([label, mae, rmse, runtime])

    headers = ["Model Label", "MAE", "RMSE", "Runtime (seconds)"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
