"""Step 4 (Q3, Q4): inverse power method with deflation."""

import numpy as np

from step5_plu import make_plu_solver


def inverse_power_deflation(L, mu=0.0, deflate_vectors=None, x0=None,
                            tol=1e-9, max_iter=1000, solver=None):
    # inverse power iteration with shift mu, deflating the given vectors
    n = L.shape[0]
    if solver is None:
        A = mu * np.eye(n) - L  # (mu I - L_sym), as in the problem statement
        solver = make_plu_solver(A)  # our own PLU factorization (step 5)

    Q = np.stack(deflate_vectors, axis=1) if deflate_vectors else np.zeros((n, 0))
    deflate = lambda x: x - Q @ (Q.T @ x) if Q.shape[1] else x

    rng = np.random.default_rng(1)
    x = deflate(rng.standard_normal(n) if x0 is None else x0.astype(float).copy())
    if np.linalg.norm(x) == 0:
        x = deflate(rng.standard_normal(n))
    x /= np.linalg.norm(x)

    lam = None
    for it in range(1, max_iter + 1):
        y = deflate(solver(x))  # re-deflate (rounding reintroduces components)
        ny = np.linalg.norm(y)
        if ny == 0:
            break
        y /= ny
        if y @ x < 0:
            y = -y
        diff = np.linalg.norm(y - x)
        x = y
        lam = float(x @ (L @ x))  # Rayleigh quotient on L
        if diff < tol:
            return lam, x, it
    return lam, x, max_iter


def compute_segmentation_eigenvectors(L, d, k, mu=1e-3, make_solver=None,
                                      tol=1e-9, max_iter=2000):
    # find z2..zk (smallest nonzero modes) via inverse power + cumulative deflation
    A = mu * np.eye(L.shape[0]) - L  # (mu I - L_sym), as in the problem statement
    # factor (mu I - L_sym) once with our own PLU (step 5), then reuse for every solve
    solver = make_plu_solver(A) if make_solver is None else make_solver(A)

    z1 = np.sqrt(d)
    z1 /= np.linalg.norm(z1)  # trivial eigenvector

    deflate_vectors, eigvals, cols = [z1], [], []
    for _ in range(k - 1):
        lam, v, _ = inverse_power_deflation(
            L, mu=mu, deflate_vectors=deflate_vectors,
            tol=tol, max_iter=max_iter, solver=solver)
        deflate_vectors.append(v)  # cumulative deflation
        eigvals.append(lam)
        cols.append(v)

    return eigvals, np.stack(cols, axis=1)


if __name__ == "__main__":
    from step1_image import prepare
    from step2_weights import build_weight_matrix
    from step3_laplacian import build_normalized_laplacian, spectral_report

    img, X, F, shape = prepare(size=50)
    W = build_weight_matrix(X, F, sigma_I=0.1, sigma_X=4.0, r=5.0)
    L, d, d_inv_sqrt = build_normalized_laplacian(W)

    k = 3
    eigvals, Z = compute_segmentation_eigenvectors(L, d, k=k, mu=1e-3)
    true_evals = spectral_report(L, d)["evals"]

    print(f"eigenvalues (inverse power + deflation), k={k}:")
    for i, lam in enumerate(eigvals, start=2):
        print(f"  z{i}: lambda = {lam:.6e}")
    print("reference (eigh):", np.array2string(true_evals[:k + 1], precision=6))

    z1 = np.sqrt(d); z1 /= np.linalg.norm(z1)
    print(f"|z2 . z1(trivial)| = {abs(Z[:, 0] @ z1):.3e}")
    if k >= 3:
        print(f"|z3 . z2|          = {abs(Z[:, 0] @ Z[:, 1]):.3e}")
    for i in range(Z.shape[1]):
        res = np.linalg.norm(L @ Z[:, i] - eigvals[i] * Z[:, i])
        print(f"residual ||L z{i + 2} - lambda z{i + 2}|| = {res:.3e}")
