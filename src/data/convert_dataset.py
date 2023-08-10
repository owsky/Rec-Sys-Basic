import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

from MF_models import all_models_type


def convert_dataset(dataset: coo_matrix | NDArray[np.float64], model: all_models_type):
    name = model.__class__.__name__

    if type(dataset) == coo_matrix and "dense" in name:
        return dataset.toarray()
    elif "sparse" in name:
        return coo_matrix(dataset)
    return dataset
