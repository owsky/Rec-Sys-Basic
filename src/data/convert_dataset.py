import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

from MF_models import all_models_type


def convert_dataset(dataset: coo_matrix | NDArray[np.float64], label: str):
    if isinstance(dataset, coo_matrix) and "dense" in label.lower():
        return dataset.toarray()
    elif "sparse" in label.lower():
        return coo_matrix(dataset)
    return dataset
