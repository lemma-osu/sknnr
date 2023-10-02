import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


def is_2d_numeric_numpy_array(arr: Any) -> bool:
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 2
        and np.issubdtype(arr.dtype, np.number)
    )


def zero_sum_rows(arr: NDArray) -> NDArray:
    return arr.sum(axis=1) <= 0.0


def zero_sum_columns(arr: NDArray) -> NDArray:
    return arr.sum(axis=0) <= 0.0


class ConstrainedOrdination(ABC):
    ZERO: float = math.sqrt(2.220446e-16)

    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self._check_inputs()
        self._transform_Y()

    @abstractmethod
    def _transform_Y(self):
        ...

    @abstractmethod
    def _transform_X(self):
        ...

    def _check_inputs(self):
        if not is_2d_numeric_numpy_array(self.X):
            msg = "X must be a 2D numeric numpy array"
            raise ValueError(msg)

        if not is_2d_numeric_numpy_array(self.Y):
            msg = "Y must be a 2D numeric numpy array"
            raise ValueError(msg)

        if self.X.shape[0] != self.Y.shape[0]:
            msg = "X and Y must have the same number of rows"
            raise ValueError(msg)

        if np.any(zero_sum_rows(self.X)):
            msg = "All row sums in X must be greater than 0"
            raise ValueError(msg)

        excluded_columns = zero_sum_columns(self.Y)
        self.Y = self.Y[:, ~excluded_columns]

    def __call__(self):
        X_scale = self._transform_X()
        Q, R = np.linalg.qr(X_scale)
        solution, _, rank, _ = np.linalg.lstsq(R, Q.T @ self.Y, rcond=None)
        Y_fit = X_scale @ solution
        U, S, _ = np.linalg.svd(Y_fit, full_matrices=False)
        self.rank = min(rank, np.sum([S > self.ZERO]))
        self.eigenvalues = np.square(S)[: self.rank]
        self.axis_weights = np.diag(np.sqrt(self.eigenvalues / self.eigenvalues.sum()))
        self.coefficients = np.linalg.lstsq(R, Q.T @ U[:, : self.rank], rcond=None)[0]
        return self

    @property
    def max_components(self):
        return self.rank

    def projector(self, n_components):
        return (
            self.coefficients[:, :n_components]
            @ self.axis_weights[:n_components, :n_components]
        )


class CCA(ConstrainedOrdination):
    def _transform_Y(self):
        normalized = self.Y / self.Y.sum()
        self.rw = normalized.sum(axis=1)
        self.cw = normalized.sum(axis=0)
        rc = np.outer(self.rw, self.cw)
        self.Y = (normalized - rc) / np.sqrt(rc)

    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0, weights=self.rw)
        X_scale = self.X - self.env_center
        return X_scale * np.sqrt(self.rw)[:, np.newaxis]


class RDA(ConstrainedOrdination):
    def _transform_Y(self):
        self.Y = self.Y - self.Y.mean(axis=0)
        self.Y /= np.sqrt(self.Y.shape[0] - 1)

    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0)
        return self.X - self.env_center
