import pandas as pd
from os import path
from scipy.sparse import coo_array, load_npz, save_npz


def load_data(dataset_path: str, limit_row: int = -1) -> coo_array:
    lim = f"_lim_{limit_row}" if limit_row != -1 else ""
    file_name = f"data{lim}.npz"
    npz_path = path.join(path.dirname(dataset_path), file_name)
    if not path.isfile(npz_path):
        print("Generating npz file")
        data = pd.read_csv(dataset_path)
        user_item_matrix = (
            data.pivot(index="User_ID", columns="Movie_ID", values="Rating")
            .fillna(0)
            .to_numpy()
        )
        user_item_matrix = coo_array(user_item_matrix[:limit_row])  # SHRINK NUM_USERS
        save_npz(npz_path, user_item_matrix)
    else:
        print("Loading stored npz file")
        user_item_matrix = load_npz(npz_path)
    return user_item_matrix
