from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


def is_2d_numeric_array(arr: NDArray) -> bool:
    return arr.ndim == 2 and np.issubdtype(arr.dtype, np.number)


def zero_sum_rows(arr: NDArray) -> NDArray:
    return arr.sum(axis=1) <= 0.0


def zero_sum_columns(arr: NDArray) -> NDArray:
    return arr.sum(axis=0) <= 0.0


class ConstrainedOrdination(ABC):
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self._check_inputs()

        self._transform_Y()
        self._transform_X()

        Q, R = np.linalg.qr(self.X)
        solution, _, rank, _ = np.linalg.lstsq(R, Q.T @ self.Y, rcond=None)
        Y_fit = self.X @ solution
        U, S, _ = np.linalg.svd(Y_fit, full_matrices=False)
        self.rank = min(rank, np.sum([S > 0.0]))
        self.eigenvalues = np.square(S)[: self.rank]
        self.axis_weights = np.diag(np.sqrt(self.eigenvalues / self.eigenvalues.sum()))
        self.coefficients = np.linalg.lstsq(R, Q.T @ U[:, : self.rank], rcond=None)[0]

    @abstractmethod
    def _transform_X(self):
        ...

    @abstractmethod
    def _transform_Y(self):
        ...

    def _check_inputs(self):
        if not is_2d_numeric_array(self.X):
            raise ValueError("X must be a 2D numeric numpy array")

        if not is_2d_numeric_array(self.Y):
            raise ValueError("Y must be a 2D numeric numpy array")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        if np.any(zero_sum_rows(self.X)):
            raise ValueError("All row sums in X must be greater than 0")

        excluded_columns = zero_sum_columns(self.Y)
        self.Y = self.Y[:, ~excluded_columns]

    @property
    def max_components(self):
        return self.rank

    def projector(self, n_components):
        return (
            self.coefficients[:, :n_components]
            @ self.axis_weights[:n_components, :n_components]
        )


class CCA(ConstrainedOrdination):
    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0, weights=self.rw)
        X_scale = self.X - self.env_center
        self.X = X_scale * np.sqrt(self.rw)[:, np.newaxis]

    def _transform_Y(self):
        normalized = self.Y / self.Y.sum()
        self.rw = normalized.sum(axis=1)
        self.cw = normalized.sum(axis=0)
        rc = np.outer(self.rw, self.cw)
        self.Y = (normalized - rc) / np.sqrt(rc)


class RDA(ConstrainedOrdination):
    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0)
        self.X = self.X - self.env_center

    def _transform_Y(self):
        self.Y = self.Y - self.Y.mean(axis=0)
        self.Y /= np.sqrt(self.Y.shape[0] - 1)
