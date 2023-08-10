import numpy as np
import pandas as pd
from os import path
from scipy.sparse import coo_array


def load_data(dataset_path: str, limit_row: int = -1):
    lim = f"lim_{limit_row}" if limit_row != -1 else ""
    file_name = f"data_{lim}.npy"
    npy_path = path.join(path.dirname(dataset_path), file_name)
    if not path.isfile(npy_path):
        print("Generating npy file")
        data = pd.read_csv(dataset_path)
        user_item_matrix = (
            data.pivot(index="User_ID", columns="Movie_ID", values="Rating")
            .fillna(0)
            .to_numpy()
        )
        user_item_matrix = user_item_matrix[0:limit_row]  # SHRINK NUM_USERS
        np.save(npy_path, user_item_matrix)
    else:
        print("Loading stored npy file")
        user_item_matrix = np.load(npy_path)
    return coo_array(user_item_matrix)
