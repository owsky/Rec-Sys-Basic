import sys
from data import load_data
from train import runtimes_all
from cross_validation import hyper_tune
from utils import memory_usage


def main():
    assert sys.argv[1], "Wrong usage: provide dataset path"
    dataset_path = sys.argv[1]
    dataset = load_data(dataset_path, limit_row=50)

    # memory_usage(dataset)

    # runtimes_all(dataset)
    hyper_tune(dataset)


if __name__ == "__main__":
    main()
