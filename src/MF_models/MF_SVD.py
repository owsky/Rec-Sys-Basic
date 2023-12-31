from numpy.typing import NDArray
import numpy as np
from .MF_base import MF_base
from scipy.sparse.linalg import svds
from scipy.sparse import coo_array


class MF_SVD(MF_base):
    def _svd_decompose(
        self, R: NDArray[np.float64] | coo_array, n_factors: int, seed: int | None
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
        R: NDArray[np.float64] | coo_array,
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

        for _ in range(epochs):
            self.lr *= lr_decay_factor

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

    def predict(self, user_idx: int, item_idex: int) -> float:
        return np.dot(self.P[user_idx, :], np.dot(self.S, self.Q[item_idex, :]))
