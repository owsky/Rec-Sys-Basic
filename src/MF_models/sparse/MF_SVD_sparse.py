from scipy.sparse import coo_matrix
import numpy as np
from ..MF_base import MF_base
from scipy.sparse.linalg import svds


class MF_SVD_sparse(MF_base):
    def _svd_decompose(self, R: coo_matrix, n_factors: int):
        U, S, V = svds(R, k=n_factors)
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
        R: coo_matrix,
        n_factors: int = 10,
        epochs: int = 80,
        lr: float = 0.00005,
        reg: float = 0.04,
    ):
        self._svd_decompose(R, n_factors)

        for _ in range(epochs):
            for u, i, r in zip(R.row, R.col, R.data):
                error = r - self.predict(u, i)  # type: ignore

                self.P[u, :] += 2 * lr * (error * self.Q[i, :] - reg * self.P[u, :])
                self.Q[i, :] += 2 * lr * (error * self.P[u, :] - reg * self.Q[i, :])

        self.R_pred = np.dot(np.dot(self.P, self.S), self.Q.T)

    def predict(self, user_idx: int, item_idex: int) -> float:
        return np.dot(self.P[user_idx, :], np.dot(self.S, self.Q[item_idex, :]))
