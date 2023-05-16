import numpy as np
from scipy.stats import f


def ftest_cor(p, q, N, cor):
    """ "Direct port of yaImpute ftest_cor in yai"""
    s = min(p, q)
    k = np.arange(s) + 1
    L = np.array([np.prod(1 - np.square(cor[i:s])) for i in range(s)])
    r = (N - s - 1) - ((abs(p - q) + 1) / 2)
    ndf = (p - k + 1) * (q - k + 1)
    u = (ndf - 2) / 4
    xx = (np.square(p - k + 1) + np.square(q - k + 1)) - 5
    t = np.array(
        [
            (
                np.sqrt(((p - k[i] + 1) ** 2 * (q - k[i] + 1) ** 2 - 4) / xx[i])
                if xx[i] > 0
                else 0
            )
            for i in range(s)
        ]
    )
    mask = t > 0
    t = t[mask]
    L = L[mask]
    u = u[mask]
    ndf = ndf[mask]
    L_invt = np.power(L, 1.0 / t)
    ddf = (r * t) - (2 * u)
    set_na = (ddf < 1.0) | (ndf < 1)
    first_na = np.nonzero(set_na)[0]
    first_na = first_na[0] if len(first_na) else len(set_na)
    set_na[first_na:] = True
    F = ((1.0 - L_invt) / L_invt) * (ddf / ndf)
    F[set_na] = np.nan
    pg_F = np.array([1.0 - f.cdf(F[i], ndf[i], ddf[i]) for i in range(F.shape[0])])
    pg_F[set_na] = np.nan
    return pg_F[~np.isnan(pg_F)]


class CCorA:
    TOLERANCE = 1e-8
    P_VAL = 0.05

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.k = min(self.X.shape[1], self.y.shape[1])

    def _apply_svd_with_tolerance(self, arr):
        u, s, v = np.linalg.svd(arr, 0)
        mask = s > self.TOLERANCE
        return u[:, mask], s[mask], v[mask][:, mask]

    @property
    def X_norm(self):
        return self.X - self.X.mean(axis=0)

    @property
    def y_norm(self):
        return self.y - self.y.mean(axis=0)

    @property
    def x_svd(self):
        return self._apply_svd_with_tolerance(self.X_norm)

    @property
    def y_svd(self):
        return self._apply_svd_with_tolerance(self.y_norm)

    @property
    def vx_ds(self):
        _, s, v = self.x_svd
        return v.T / s

    @property
    def vy_ds(self):
        _, s, v = self.y_svd
        return v.T / s

    @property
    def svd(self):
        ux, uy = self.x_svd[0], self.y_svd[0]
        return np.linalg.svd(ux.T.dot(uy), full_matrices=False)

    @property
    def cancorr(self):
        s = self.svd[1]
        return np.array([max(0, min(s[i], 1)) for i in range(len(s))])

    @property
    def cscal(self):
        canonical_coef = self.vx_ds @ self.svd[0][:, 0]
        return 1.0 / np.std(self.X_norm @ canonical_coef, ddof=1)

    @property
    def x_coef(self):
        return self.vx_ds @ self.svd[0][:, : self.k] * self.cscal

    @property
    def y_coef(self):
        return self.vy_ds @ self.svd[2].T[:, : self.k] * self.cscal

    @property
    def f_test(self):
        return ftest_cor(
            self.y_coef.shape[0],
            self.x_coef.shape[0],
            self.y_norm.shape[0],
            self.cancorr,
        )

    @property
    def n_vec(self):
        return max(1, len(self.f_test) - sum(self.f_test > self.P_VAL))

    @property
    def projector(self):
        return self.x_coef[:, : self.n_vec] @ np.diag(self.cancorr[: self.n_vec])
