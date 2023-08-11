import numpy as np
from os import path
from scipy.sparse import coo_array, vstack
import h5py


def load_data_sparse(dataset_path: str, limit_row: int = -1):
    lim = f"lim_{limit_row}" if limit_row != -1 else ""
    file_name = f"data_{lim}.h5"
    h5_path = path.join(path.dirname(dataset_path), file_name)
    if not path.isfile(h5_path):
        print("Generating npy file")
        user_item_matrix = []
        with open(dataset_path) as csv_file:
            for i, row in enumerate(csv_file):
                if i == limit_row:
                    break
                data = np.fromstring(row, sep=",")
                user_item_matrix.append(coo_array(data))

        user_item_matrix = vstack(user_item_matrix)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("Netflix dataset", data=user_item_matrix)
            f.close()
    else:
        print("Loading stored npy file")
        with h5py.File(h5_path, "r") as f:
            dataset = f["Netflix dataset"]
            user_item_matrix = coo_array(dataset)
    return coo_array(user_item_matrix)
