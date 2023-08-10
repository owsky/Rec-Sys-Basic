from pympler.asizeof import asizeof
from utils import sparsity
from scipy.sparse import coo_array, coo_matrix
import matplotlib.pyplot as plt
import pandas as pd


def memory_usage(dataset: coo_array):
    dataset_arr = dataset.toarray()
    spar = sparsity(dataset_arr)
    print(f"Sparsity of dataset: {spar:.3f}")

    df_mem = asizeof(pd.DataFrame(dataset_arr)) / 1024 / 1024 / 1024
    coo_array_mem = asizeof(coo_matrix(dataset)) / 1024 / 1024 / 1024
    dense_mem = asizeof(dataset_arr) / 1024 / 1024 / 1024
    csr_mem = asizeof(dataset.tocsr()) / 1024 / 1024 / 1024

    data_structures = ["Pandas DataFrame", "Numpy Array", "COO Matrix", "CSR Matrix"]
    memory_usage = [df_mem, dense_mem, coo_array_mem, csr_mem]

    plt.bar(data_structures, memory_usage, color=["purple", "red", "green", "orange"])

    plt.xlabel("Data Structures")
    plt.ylabel("Memory Usage (gigabytes)")
    plt.title("Memory Usage of Data Structures")

    # Show the plot
    plt.show()
