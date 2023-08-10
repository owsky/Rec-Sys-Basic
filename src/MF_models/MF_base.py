import math
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import List, Callable
from scipy.sparse import coo_matrix, dok_array, csr_matrix


R_type = NDArray[np.float64] | coo_matrix | dok_array | csr_matrix


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
        num_users, num_items = user_item_matrix.shape
        errors = []
        for user_id in range(num_users):
            for item_id in range(num_items):
                if user_item_matrix[user_id, item_id] != 0:
                    predicted_rating = self.predict(user_id, item_id)
                    true_rating = user_item_matrix[user_id, item_id]
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
