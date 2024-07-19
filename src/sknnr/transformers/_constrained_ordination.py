from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


def is_2d_numeric_array(arr: NDArray) -> bool:
    """Verify that arr is a 2D numeric array."""
    return arr.ndim == 2 and np.issubdtype(arr.dtype, np.number)


def zero_sum_vectors(arr: NDArray, *, axis: int) -> NDArray:
    """Find any rows or columns in arr that sum to 0."""
    return arr.sum(axis=axis) <= 0.0


class ConstrainedOrdination(ABC):
    """Superclass for constrained ordination methods."""

    def __init__(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        X, Y = self._check_inputs(X, Y)
        self._set_initialization_attributes(X, Y)

        Q, R = np.linalg.qr(self.X)
        solution, _, rank, _ = np.linalg.lstsq(R, Q.T @ self.Y, rcond=None)
        Y_fit = self.X @ solution
        U, S, _ = np.linalg.svd(Y_fit, full_matrices=False)
        self.rank = min(rank, np.sum([S > 0.0]))
        self.eigenvalues = np.square(S)[: self.rank]
        self.axis_weights = np.diag(np.sqrt(self.eigenvalues / self.eigenvalues.sum()))
        self.coefficients = np.linalg.lstsq(R, Q.T @ U[:, : self.rank], rcond=None)[0]

    @abstractmethod
    def _set_initialization_attributes(self, X: NDArray, Y: NDArray) -> None:
        """Set method-specific instance-level attributes needed for ordination,
        including possibly modifying input X and Y arrays. Must be implemented
        by subclasses."""
        ...

    @staticmethod
    def _check_inputs(X: NDArray, Y: NDArray) -> tuple[NDArray, NDArray]:
        """Verify that X and Y are valid inputs in preparation for ordination."""
        if not is_2d_numeric_array(X):
            raise ValueError("X must be a 2D numeric numpy array")

        if not is_2d_numeric_array(Y):
            raise ValueError("Y must be a 2D numeric numpy array")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        if np.any(zero_sum_vectors(X, axis=1)):
            raise ValueError("All row sums in X must be greater than 0")

        excluded_columns = zero_sum_vectors(Y, axis=0)
        return X, Y[:, ~excluded_columns]

    @property
    def max_components(self) -> int:
        """Return the maximum number of components that can be extracted
        from the ordination."""
        return self.rank

    def projector(self, n_components: int) -> NDArray:
        """Return the projection matrix for the first n_components."""
        return (
            self.coefficients[:, :n_components]
            @ self.axis_weights[:n_components, :n_components]
        )


class CCA(ConstrainedOrdination):
    def _set_initialization_attributes(self, X: NDArray, Y: NDArray) -> None:
        normalized = Y / Y.sum()
        self.rw = normalized.sum(axis=1)
        self.cw = normalized.sum(axis=0)
        rc = np.outer(self.rw, self.cw)
        self.Y = (normalized - rc) / np.sqrt(rc)

        self.env_center = np.average(X, axis=0, weights=self.rw)
        self.X = (X - self.env_center) * np.sqrt(self.rw)[:, np.newaxis]


class RDA(ConstrainedOrdination):
    def _set_initialization_attributes(self, X: NDArray, Y: NDArray) -> None:
        self.env_center = np.average(X, axis=0)
        self.X = X - self.env_center
        self.Y = Y - Y.mean(axis=0)
        self.Y /= np.sqrt(self.Y.shape[0] - 1)
