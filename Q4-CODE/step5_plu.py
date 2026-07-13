"""Step 5 (Q5): PLU decomposition with partial pivoting + triangular solves, implemented from scratch."""

import numpy as np


def plu_decomposition(A):
    # Gaussian elimination with partial pivoting -> P A = L U
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n):
        # partial pivoting: bring the largest |entry| of column k to the diagonal
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            P[[k, p], :] = P[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]
        # eliminate the entries below the pivot
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return P, L, U


def forward_substitution(L, b):
    # solve L y = b for a lower-triangular L with unit diagonal
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_substitution(U, y):
    # solve U x = y for an upper-triangular U
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def make_plu_solver(A):
    # factor A = P^T L U once; afterwards each solve is just two triangular solves
    P, L, U = plu_decomposition(A)

    def solve(b):
        pb = P @ np.asarray(b, dtype=float)
        return back_substitution(U, forward_substitution(L, pb))

    return solve


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    P, L, U = plu_decomposition(A)
    print(f"||P A - L U|| = {np.linalg.norm(P @ A - L @ U):.3e}")

    b = rng.standard_normal(6)
    x = make_plu_solver(A)(b)
    print(f"||A x - b||   = {np.linalg.norm(A @ x - b):.3e}")
