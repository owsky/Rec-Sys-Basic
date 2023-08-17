import math
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import List, Callable
from scipy.sparse import coo_array


R_type = NDArray[np.float64] | coo_array


class MF_base(ABC):
    def __init__(self):
        self.P: NDArray[np.float64] = np.array([])
        self.Q: NDArray[np.float64] = np.array([])

    @abstractmethod
    def fit(
        self,
        R: R_type,
        n_factors: int = 10,
        lr: float = 0.005,
        epochs: int = 20,
        reg: float = 0.02,
        batch_size: int = 0,
        seed: int | None = None,
    ):
        pass

    def predict(self, u: int, i: int) -> float:
        if self.P.size == 0 or self.Q.size == 0:
            raise Exception("Model untrained, invoke fit before predicting")
        return np.dot(self.P[u, :], self.Q[i, :].T)

    def _compute_prediction_errors(
        self,
        user_item_matrix: NDArray[np.float64],
        error_function: Callable[[float, float], float],
    ) -> List[float]:
        errors = []
        users, items = np.nonzero(user_item_matrix)
        for user_id, item_id in zip(users, items):
            if user_item_matrix[user_id, item_id] != 0:
                predicted_rating = self.predict(user_id, item_id)
                true_rating = float(user_item_matrix[user_id, item_id])
                errors.append(error_function(true_rating, predicted_rating))
        return errors

    def accuracy_mae(self, user_item_matrix: NDArray[np.float64]) -> float:
        errors = self._compute_prediction_errors(
            user_item_matrix, lambda t, p: abs(t - p)
        )
        mae = float(np.mean(errors))
        return mae

    def accuracy_rmse(self, user_item_matrix: NDArray[np.float64]) -> float:
        errors = self._compute_prediction_errors(
            user_item_matrix, lambda t, p: (t - p) ** 2
        )
        rmse = math.sqrt(np.mean(errors))
        return rmse

    def _clip_gradients(
        self, gradient: NDArray[np.float64], max_grad_norm: float | None
    ):
        if max_grad_norm is not None:
            norm = np.linalg.norm(gradient)
            if norm > max_grad_norm:
                gradient *= max_grad_norm / norm

    def _update_features(
        self,
        errors: NDArray[np.float64],
        user: int,
        item: int,
        reg: float,
        lr: float,
        max_grad_norm: float | None,
    ):
        grad_P = 2 * lr * (errors * self.Q[item, :] - reg * self.P[user, :])
        grad_Q = 2 * lr * (errors * self.P[user, :] - reg * self.Q[item, :])

        self._clip_gradients(grad_P, max_grad_norm)
        self._clip_gradients(grad_Q, max_grad_norm)

        self.P[user, :] += grad_P
        self.Q[item, :] += grad_Q
