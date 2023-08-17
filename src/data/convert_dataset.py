from scipy.sparse import coo_array
from MF_models import R_type


def convert_dataset(dataset: R_type, label: str) -> R_type:
    if isinstance(dataset, coo_array) and "dense" in label.lower():
        return dataset.toarray()
    elif "sparse" in label.lower():
        return coo_array(dataset)
    return dataset
