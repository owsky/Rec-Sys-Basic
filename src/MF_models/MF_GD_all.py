import numpy as np
from numpy.typing import NDArray
from .MF_base import MF_base
from scipy.sparse import coo_matrix


class MF_GD_all(MF_base):
    def _clip_gradients(self, gradient):
        if self.max_grad_norm is not None:
            norm = np.linalg.norm(gradient)
            if norm > self.max_grad_norm:
                gradient *= self.max_grad_norm / norm

    def _update_features_sgd(self, error: NDArray[np.float64], u: int, i: int):
        grad_P = 2 * self.lr * (error * self.Q[i, :] - self.reg * self.P[u, :])
        grad_Q = 2 * self.lr * (error * self.P[u, :] - self.reg * self.Q[i, :])

        self._clip_gradients(grad_P)
        self._clip_gradients(grad_Q)

        self.P[u, :] += grad_P
        self.Q[i, :] += grad_Q

    def _update_features_batch(
        self, errors: NDArray[np.float64], u: NDArray[np.int64], i: NDArray[np.int64]
    ):
        grad_P = (
            2
            * self.lr
            * (errors[:, np.newaxis] * self.Q[i, :] - self.reg * self.P[u, :])
        )
        grad_Q = (
            2
            * self.lr
            * (errors[:, np.newaxis] * self.P[u, :] - self.reg * self.Q[i, :])
        )

        self._clip_gradients(grad_P)
        self._clip_gradients(grad_Q)

        self.P[u, :] += grad_P
        self.Q[i, :] += grad_Q

    def sgd(self, R: NDArray[np.float64] | coo_matrix):
        if isinstance(R, coo_matrix):
            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor
                for u, i, r in zip(R.row, R.col, R.data):
                    error = r - self.predict(u, i)
                    self._update_features_sgd(error, u, i)
        else:
            users, items = np.nonzero(R)
            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor
                for u, i in zip(users, items):
                    error = R[u, i] - self.predict(u, i)

                    self._update_features_sgd(error, u, i)

    def mini_batch(self, R: NDArray[np.float64] | coo_matrix):
        if isinstance(R, coo_matrix):
            data = list(zip(R.row, R.col, R.data))

            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor
                np.random.shuffle(data)

                for i in range(0, len(data), self.batch_size):
                    batch = data[i : i + self.batch_size]

                    users = np.array([user for user, _, _ in batch])
                    items = np.array([item for _, item, _ in batch])
                    ratings = np.array([rating for _, _, rating in batch])

                    errors = ratings - np.sum(
                        self.P[users, :] * self.Q[items, :], axis=1
                    )
                    self._update_features_batch(errors, users, items)
        else:
            nonzero_indices = np.transpose(np.nonzero(R))
            total_samples = len(nonzero_indices)
            shuffled_indices = np.random.permutation(total_samples)

            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor

                for i in range(0, len(nonzero_indices), self.batch_size):
                    batch_indices = shuffled_indices[i : i + self.batch_size]
                    if len(batch_indices) < self.batch_size:
                        break
                    batch_user_indices, batch_item_indices = (
                        nonzero_indices[batch_indices, 0],
                        nonzero_indices[batch_indices, 1],
                    )

                    batch_R = R[batch_user_indices, batch_item_indices]

                    predicted_ratings = np.sum(
                        self.P[batch_user_indices, :] * self.Q[batch_item_indices, :],
                        axis=1,
                    )

                    errors = batch_R - predicted_ratings
                    self._update_features_batch(
                        errors, batch_user_indices, batch_item_indices
                    )

    def process_batch(self, R: NDArray[np.float64] | coo_matrix):
        if isinstance(R, coo_matrix):
            non_zero_row_indices, non_zero_col_indices = R.nonzero()
            non_zero_values = R.data

            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor
                errors = non_zero_values - np.sum(
                    self.P[non_zero_row_indices, :] * self.Q[non_zero_col_indices, :],
                    axis=1,
                )

                self._update_features_batch(
                    errors, non_zero_row_indices, non_zero_col_indices
                )
        else:
            non_zero = np.nonzero(R)
            for _ in range(self.epochs):
                self.lr *= self.lr_decay_factor
                errors = R[non_zero] - np.sum(
                    self.P[non_zero[0], :] * self.Q[non_zero[1], :], axis=1
                )
                self._update_features_batch(errors, non_zero[0], non_zero[1])

    def fit(
        self,
        R: NDArray[np.float64] | coo_matrix,
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
