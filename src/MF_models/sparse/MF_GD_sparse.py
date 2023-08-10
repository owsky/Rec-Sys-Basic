import numpy as np
from ..MF_base import MF_base
from scipy.sparse import coo_matrix


class MF_GD_sparse(MF_base):
    def sgd(self, R: coo_matrix, epochs: int, lr: float, reg: float):
        print("Training with stochastic")
        for _ in range(epochs):
            for u, i, r in zip(R.row, R.col, R.data):
                error = r - self.predict(u, i)
                self.P[u, :] += 2 * lr * (error * self.Q[i, :] - reg * self.P[u, :])
                self.Q[i, :] += 2 * lr * (error * self.P[u, :] - reg * self.Q[i, :])

    def mini_batch(
        self,
        R: coo_matrix,
        learning_rate: float,
        regularization_rate: float,
        num_epochs: int,
        batch_size: int,
    ):
        data = list(zip(R.row, R.col, R.data))

        for _ in range(num_epochs):
            # Shuffle the data
            np.random.shuffle(data)

            # Mini-batch updates
            for i in range(0, len(data), batch_size):
                # Extract a mini-batch of data
                batch = data[i : i + batch_size]
                batch_size = len(
                    batch
                )  # Actual batch size (might be smaller at the end)

                # Convert the batch data to arrays
                users = np.array([user for user, _, _ in batch])
                items = np.array([item for _, item, _ in batch])
                ratings = np.array([rating for _, _, rating in batch])

                # Compute the gradient for the mini-batch
                errors = ratings - np.sum(self.P[users, :] * self.Q[items, :], axis=1)
                self.P[users, :] += (
                    2 * learning_rate * (errors[:, np.newaxis] * self.Q[items, :])
                    - regularization_rate * self.P[users, :]
                )
                self.Q[items, :] += (
                    2 * learning_rate * (errors[:, np.newaxis] * self.P[users, :])
                    - regularization_rate * self.Q[items, :]
                )

    def process_batch(self, R: coo_matrix, epochs: int, lr: float, reg: float):
        print("Training with batch")

        # Extract non-zero elements from R
        non_zero_row_indices, non_zero_col_indices = R.nonzero()
        non_zero_values = R.data

        for _ in range(epochs):
            errors = non_zero_values - np.sum(
                self.P[non_zero_row_indices, :] * self.Q[non_zero_col_indices, :],
                axis=1,
            )
            self.P[non_zero_row_indices, :] += (
                2
                * lr
                * (
                    errors[:, np.newaxis] * self.Q[non_zero_col_indices, :]
                    - reg * self.P[non_zero_row_indices, :]
                )
            )
            self.Q[non_zero_col_indices, :] += (
                2
                * lr
                * (
                    errors[:, np.newaxis] * self.P[non_zero_row_indices, :]
                    - reg * self.Q[non_zero_col_indices, :]
                )
            )

    def fit(
        self,
        R: coo_matrix,
        n_factors: int = 10,
        epochs: int = 20,
        lr: float = 0.009,
        reg: float = 0.002,
        batch_size: int = 8,
    ):
        num_users, num_items = R.shape

        self.P = np.random.normal(loc=0, scale=0.1, size=(num_users, n_factors))
        self.Q = np.random.normal(loc=0, scale=0.1, size=(num_items, n_factors))

        # stochastic
        if batch_size == 1:
            self.sgd(R, epochs, lr, reg)
        # batch
        elif batch_size == num_users:
            self.process_batch(R, epochs, lr, reg)
        # mini batch
        else:
            self.mini_batch(R, lr, reg, epochs, batch_size)
