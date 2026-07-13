"""Step 2 (Q1): build the symmetric similarity matrix W."""

import numpy as np
import scipy.sparse as sp


def build_weight_matrix(X, F, sigma_I=0.1, sigma_X=4.0, r=5.0, dense=True):
    # W_ij = exp(-||Fi-Fj||^2/sigma_I^2) * exp(-||Xi-Xj||^2/sigma_X^2) for ||Xi-Xj|| < r, else 0
    # squared pairwise distances via ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    sqX = np.sum(X ** 2, axis=1)
    dX2 = sqX[:, None] + sqX[None, :] - 2.0 * (X @ X.T)
    sqF = np.sum(F ** 2, axis=1)
    dF2 = sqF[:, None] + sqF[None, :] - 2.0 * (F @ F.T)
    np.clip(dX2, 0.0, None, out=dX2)  # tiny negatives can appear from rounding

    W = np.exp(-dF2 / sigma_I ** 2) * np.exp(-dX2 / sigma_X ** 2)
    W[dX2 >= r ** 2] = 0.0  # keep only spatial neighbors within radius r
    np.fill_diagonal(W, 0.0)  # W_ii = 0
    return W if dense else sp.csr_matrix(W)


def symmetry_error(W):
    # ||W - W^T||_F
    if sp.issparse(W):
        diff = W - W.T
        return float(np.sqrt((diff.multiply(diff)).sum()))
    return float(np.linalg.norm(W - W.T, ord="fro"))


def sparsity(W):
    # fraction of nonzero entries
    nnz = W.nnz if sp.issparse(W) else int(np.count_nonzero(W))
    return nnz / (W.shape[0] ** 2)


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_image import prepare

    sigma_I, sigma_X, r = 0.1, 4.0, 5.0
    img, X, F, shape = prepare(size=50)
    W = build_weight_matrix(X, F, sigma_I, sigma_X, r)

    print(f"params: sigma_I={sigma_I}, sigma_X={sigma_X}, r={r}")
    print(f"W shape : {W.shape}, range [{W.min():.4f}, {W.max():.4f}]")
    print(f"max|diag|: {np.abs(np.diag(W)).max():.2e}")
    print(f"nonzeros: {sparsity(W) * 100:.2f}%")
    print(f"||W - W^T||_F = {symmetry_error(W):.3e}")

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.spy(W, markersize=0.2)
    plt.title("sparsity pattern of W")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "step2_W_spy.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step2_W_spy.png')}")
