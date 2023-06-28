import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def zero_sum_rows(arr: NDArray) -> NDArray:
    return arr.sum(axis=1) <= 0.0


def zero_sum_columns(arr: NDArray) -> NDArray:
    return arr.sum(axis=0) <= 0.0


# Corresponds to vegan function "initCA" in ordConstrained.R
@dataclass
class InitCA:
    Y_: NDArray
    method: str = "CA"

    @property
    def tot(self):
        return np.sum(self.Y_)

    def _normalized(self):
        return self.Y_ / self.tot

    @property
    def rw(self):
        return np.sum(self._normalized(), axis=1)

    @property
    def cw(self):
        return np.sum(self._normalized(), axis=0)

    @property
    def Y(self):
        rc = np.outer(self.rw, self.cw)
        return (self._normalized() - rc) / np.sqrt(rc)


# Corresponds to vegan function "ordHead" in ordConstrained.R
class OrdHead:
    def __init__(self, transform):
        self.transform = transform

    @property
    def tot_chi(self):
        Y = self.transform.Y
        return np.sum(Y * Y)

    @property
    def Ybar(self):
        return self.transform.Y

    @property
    def grand_total(self):
        return self.transform.tot

    @property
    def rowsum(self):
        return self.transform.rw

    @property
    def colsum(self):
        return self.transform.cw


class ConstrainedOrdination:
    Y_TRANSFORM = {
        "CCA": InitCA,
    }
    ZERO: float = math.sqrt(2.220446e-16)

    @property
    def transform(self):
        return self.Y_TRANSFORM[self.method](self.Y)

    @property
    def head(self):
        return OrdHead(self.transform)

    @property
    def dist_based(self) -> bool:
        return self.transform.method == "DISTBASED"

    @property
    def rw(self):
        return self.transform.rw

    @property
    def cw(self):
        return self.transform.cw

    @property
    def env_center(self):
        return np.average(self.X, axis=0, weights=self.rw)

    @property
    def X_scale(self):
        X = self.X - self.env_center
        return X * np.sqrt(self.rw)[:, np.newaxis]

    @property
    def qr(self):
        return np.linalg.qr(self.X_scale)

    @property
    def Q(self):
        return self.qr[0]

    @property
    def R(self):
        return self.qr[1]

    @property
    def _least_squares(self):
        Q, R = self.qr
        right = Q.T @ self.transform.Y
        return np.linalg.lstsq(R, right, rcond=None)

    @property
    def Y_fit(self):
        return self.X_scale @ self._least_squares[0]

    @property
    def sol(self):
        return np.linalg.svd(self.Y_fit, full_matrices=False)

    @property
    def rank(self):
        ls_rank = self._least_squares[2]
        pos_s = np.sum([self.sol[1] > self.ZERO])
        return min(ls_rank, pos_s)

    @property
    def _U(self):
        return self.sol[0][:, : self.rank]

    @property
    def S(self):
        return np.square(self.sol[1])[: self.rank]

    @property
    def _V(self):
        return self.sol[2].T[:, : self.rank]

    @property
    def U(self):
        return self._U / np.sqrt(self.rw)[:, None]

    @property
    def V(self):
        return self._V / np.sqrt(self.cw)[:, None]

    @property
    def _WA(self):
        x = np.diag(1.0 / np.sqrt(self.S))
        return self.transform.Y.dot(self._V).dot(x)

    @property
    def WA(self):
        return self._WA / np.sqrt(self.rw)[:, None]

    @property
    def biplot_scores(self):
        a = 1.0 / np.sqrt(np.square(self.X_scale).sum(axis=0))
        b = self.X_scale.T.dot(self._U)
        return a[:, None] * b

    @property
    def Y_resid(self):
        return self.transform.Y - self.Y_fit

    @property
    def max_components(self):
        return self.rank

    @property
    def eigenvalues(self):
        return self.S

    @property
    def axis_weights(self):
        return np.diag(np.sqrt(self.S / self.S.sum()))

    @property
    def coefficients(self):
        Q, R = self.qr
        right = Q.T @ self._U
        return np.linalg.lstsq(R, right, rcond=None)[0]

    def projector(self, n_components=None):
        if n_components is None:
            return self.coefficients @ self.axis_weights
        return (
            self.coefficients[:, :n_components]
            @ self.axis_weights[:n_components, :n_components]
        )

    @property
    def species_scores(self):
        return np.multiply(self.V, np.sqrt(self.S))

    @property
    def site_lc_scores(self):
        return self.U

    @property
    def site_wa_scores(self):
        return self.WA

    @property
    def species_tolerances(self):
        xi = self.site_lc_scores
        uk = self.species_scores
        xiuk = np.zeros((uk.shape[0], xi.shape[0], xi.shape[1]), dtype=np.float64)
        for i, s in enumerate(uk):
            xiuk[i] = xi - s
        y = self.Y.T
        y_xiuk_sqr = np.zeros((uk.shape[0], uk.shape[1]), dtype=np.float64)
        for i in range(y.shape[0]):
            y_xiuk_sqr[i] = y[i] @ np.square(xiuk[i])
        return np.sqrt(y_xiuk_sqr / y.sum(axis=1).reshape(-1, 1))


class CCA(ConstrainedOrdination):
    method = "CCA"
    inertia = "scaled Chi-square"

    # X is environment
    # Y is species
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if np.any(zero_sum_rows(self.Y)):
            msg = "All row sums must be greater than 0"
            raise ValueError(msg)
        self.exclude_Y = zero_sum_columns(self.Y)
        self.Y = self.Y[:, ~self.exclude_Y]
        # Remove excluded species from CCA v and CA v matrices
