from .cross_validate import cross_validate

from scipy.sparse import coo_array


def hyper_tune(dataset: coo_array):
    R = dataset.toarray()

    model_classes = []

    n_factors_range = [10, 12, 15]
    lr_range = [0.00005, 0.00003, 0.00001]
    epochs_range = [20, 30, 40, 50, 70, 80]
    reg_range = [0.01, 0.02, 0.03, 0.04]
    batch_size_range = [600, 1000, 1500]

    for index, model_class in enumerate(model_classes):
        print(
            f"{index+1}/{len(model_classes)} - Looking for best params for {model_class.__name__}"
        )
        best_params = cross_validate(
            model_cls=model_class,
            R=R,
            n_factors_range=n_factors_range,
            lr_range=lr_range,
            epochs_range=epochs_range,
            reg_range=reg_range,
            n_folds=3,
            n_jobs=-1,
            batch_size_range=batch_size_range,
        )
        print(f"Best params for {model_class.__name__}: {best_params}")
