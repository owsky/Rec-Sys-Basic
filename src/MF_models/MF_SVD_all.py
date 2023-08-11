from numpy.typing import NDArray
import numpy as np
from .MF_base import MF_base
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix


class MF_SVD_all(MF_base):
    def _clip_gradients(self, gradient, max_grad_norm):
        if max_grad_norm is not None:
            norm = np.linalg.norm(gradient)
            if norm > max_grad_norm:
                gradient *= max_grad_norm / norm

    def _update_gradients(self, error, u, i):
        grad_P = 2 * self.lr * (error * self.Q[i, :] - self.reg * self.P[u, :])
        grad_Q = 2 * self.lr * (error * self.P[u, :] - self.reg * self.Q[i, :])

        self._clip_gradients(grad_P, self.max_grad_norm)
        self._clip_gradients(grad_Q, self.max_grad_norm)

        self.P[u, :] += grad_P
        self.Q[i, :] += grad_Q

    def _svd_decompose(
        self, R: NDArray[np.float64] | coo_matrix, n_factors: int, seed: int | None
    ):
        U, S, V = (
            svds(R, k=n_factors, random_state=seed)
            if seed is not None
            else svds(R, k=n_factors)
        )
        if U is None or V is None:
            raise RuntimeError(
                "Something unexpected occurred during singular value decomposition"
            )
        else:
            self.P = U
            self.Q = V.T
            self.S = np.diag(S)

    def fit(
        self,
        R: NDArray[np.float64] | coo_matrix,
        n_factors: int = 10,
        epochs: int = 80,
        lr: float = 0.00005,
        reg: float = 0.04,
        batch_size: int = 1,
        seed: int | None = None,
        lr_decay_factor: float = 1.0,
        max_grad_norm: float | None = 1.0,
    ):
        self.lr = lr
        self.reg = reg
        self.max_grad_norm = max_grad_norm
        self._svd_decompose(R, n_factors, seed)

        if isinstance(R, coo_matrix):
            for _ in range(epochs):
                lr *= lr_decay_factor
                for u, i, r in zip(R.row, R.col, R.data):
                    error = r - self.predict(u, i)
                    self._update_gradients(error, u, i)
        else:
            users, items = np.nonzero(R)
            for _ in range(epochs):
                lr *= lr_decay_factor
                for u, i in zip(users, items):
                    error = R[u, i] - self.predict(u, i)
                    self._update_gradients(error, u, i)

    def predict(self, user_idx: int, item_idex: int) -> float:
        return np.dot(self.P[user_idx, :], np.dot(self.S, self.Q[item_idex, :]))
