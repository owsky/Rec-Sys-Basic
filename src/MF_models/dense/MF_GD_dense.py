import numpy as np
from numpy.typing import NDArray
from ..MF_base import MF_base


class MF_GD_dense(MF_base):
    def sgd(self, R, epochs, lr, reg):
        print("Training with stochastic")
        users, items = np.nonzero(R)
        for _ in range(epochs):
            for u, i in zip(users, items):
                error = R[u, i] - self.predict(u, i)
                self.P[u, :] += 2 * lr * (error * self.Q[i, :] - reg * self.P[u, :])
                self.Q[i, :] += 2 * lr * (error * self.P[u, :] - reg * self.Q[i, :])

    def mini_batch(self, R, learning_rate, regularization_rate, num_epochs, batch_size):
        # Extract the indices of non-zero elements in the ratings matrix R
        nonzero_indices = np.transpose(np.nonzero(R))

        for _ in range(num_epochs):
            # Shuffle the indices of non-zero ratings
            np.random.shuffle(nonzero_indices)

            for i in range(0, len(nonzero_indices), batch_size):
                # Select a mini-batch of non-zero ratings
                batch_indices = nonzero_indices[i : i + batch_size]
                batch_user_indices, batch_item_indices = (
                    batch_indices[:, 0],
                    batch_indices[:, 1],
                )

                # Extract the mini-batch of ratings for the selected users and items
                batch_R = R[batch_user_indices, batch_item_indices]

                # Compute the predicted ratings for the mini-batch
                predicted_ratings = np.sum(
                    self.P[batch_user_indices, :] * self.Q[batch_item_indices, :],
                    axis=1,
                )

                errors = batch_R - predicted_ratings
                self.P[batch_user_indices, :] += (
                    2
                    * learning_rate
                    * (errors[:, np.newaxis] * self.Q[batch_item_indices, :])
                    - regularization_rate * self.P[batch_user_indices, :]
                )
                self.Q[batch_item_indices, :] += (
                    2
                    * learning_rate
                    * (errors[:, np.newaxis] * self.P[batch_user_indices, :])
                    - regularization_rate * self.Q[batch_item_indices, :]
                )

    def process_batch(self, R, epochs, lr, reg):
        print("Training with batch")
        non_zero = np.nonzero(R)
        for _ in range(epochs):
            errors = R[non_zero] - np.sum(
                self.P[non_zero[0], :] * self.Q[non_zero[1], :], axis=1
            )
            self.P[non_zero[0], :] += (
                2
                * lr
                * (
                    errors[:, np.newaxis] * self.Q[non_zero[1], :]
                    - reg * self.P[non_zero[0], :]
                )
            )
            self.Q[non_zero[1], :] += (
                2
                * lr
                * (
                    errors[:, np.newaxis] * self.P[non_zero[0], :]
                    - reg * self.Q[non_zero[1], :]
                )
            )

    def fit(
        self,
        R: NDArray[np.float64],
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
