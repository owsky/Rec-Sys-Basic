import numpy as np
from numpy.typing import NDArray
from .MF_base import MF_base
from scipy.sparse import coo_array


class MF_GD(MF_base):
    def sgd(self, R: NDArray[np.float64] | coo_array):
        for _ in range(self.epochs):
            self.lr *= self.lr_decay_factor

            it: zip[tuple[int, int]] | zip[tuple[int, int, int]]
            if isinstance(R, coo_array):
                it = zip(R.row, R.col, R.data)
            else:
                users, items = np.nonzero(R)
                it = zip(users, items)

            for tup in it:
                if isinstance(R, coo_array) and len(tup) == 3:
                    u, i, r = tup
                elif not isinstance(R, coo_array) and len(tup) == 2:
                    u, i = tup
                    r = R[u, i]
                else:
                    raise RuntimeError("Something wrong occurred")
                error = r - self.predict(u, i)
                self._update_features(
                    errors=error,
                    user=u,
                    item=i,
                    reg=self.reg,
                    lr=self.lr,
                    max_grad_norm=self.max_grad_norm,
                )

    def mini_batch(self, R: NDArray[np.float64] | coo_array):
        data = (
            list(zip(R.row, R.col, R.data))
            if isinstance(R, coo_array)
            else np.transpose(np.nonzero(R))
        )

        for _ in range(self.epochs):
            self.lr *= self.lr_decay_factor
            np.random.shuffle(data)

            for i in range(0, len(data), self.batch_size):
                if isinstance(R, coo_array):
                    batch = data[i : i + self.batch_size]
                    users = np.array([user for user, _, _ in batch])
                    items = np.array([item for _, item, _ in batch])
                    ratings = np.array([rating for _, _, rating in batch])
                else:
                    batch_indices = data[i : i + self.batch_size]
                    users, items = (batch_indices[:, 0], batch_indices[:, 1])  # type: ignore
                    ratings = R[users, items]

                predictions = np.sum(self.P[users, :] * self.Q[items, :], axis=1)
                errors = ratings - predictions
                self._update_features(
                    errors=errors[:, np.newaxis],
                    user=users,
                    item=items,
                    reg=self.reg,
                    lr=self.lr,
                    max_grad_norm=self.max_grad_norm,
                )

    def process_batch(self, R: NDArray[np.float64] | coo_array):
        if isinstance(R, coo_array):
            users, items = R.nonzero()
            ratings = R.data
        else:
            users, items = np.nonzero(R)
            ratings = R[users, items]

        for _ in range(self.epochs):
            self.lr *= self.lr_decay_factor
            errors = ratings - np.sum(self.P[users, :] * self.Q[items, :], axis=1)

            self._update_features(
                errors=errors[:, np.newaxis],
                user=users,
                item=items,
                reg=self.reg,
                lr=self.lr,
                max_grad_norm=self.max_grad_norm,
            )

    def fit(
        self,
        R: NDArray[np.float64] | coo_array,
        n_factors: int = 10,
        epochs: int = 20,
        lr: float = 0.009,
        reg: float = 0.002,
        batch_size: int = 8,
        seed: int | None = None,
        lr_decay_factor: float = 0.9,
        max_grad_norm: float | None = 1.0,
    ):
        num_users, num_items = R.shape
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.max_grad_norm = max_grad_norm
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size

        if seed is not None:
            np.random.seed(seed=seed)

        if batch_size == -1:
            self.P = np.random.random((num_users, n_factors))
            self.Q = np.random.random((num_items, n_factors))
        else:
            self.P = np.random.normal(loc=0, scale=0.1, size=(num_users, n_factors))
            self.Q = np.random.normal(loc=0, scale=0.1, size=(num_items, n_factors))

        # stochastic
        if batch_size == 1:
            self.sgd(R=R)
        # batch
        elif batch_size == num_users or batch_size == -1:
            self.process_batch(R=R)
        # mini batch
        else:
            self.mini_batch(R=R)
