"""Step 3 (Q2): build the symmetric normalized Laplacian L_sym."""

import numpy as np


def build_normalized_laplacian(W):
    # L_sym = I - D^{-1/2} W D^{-1/2}
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))  # guard isolated nodes
    L = np.eye(n) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)  # enforce exact symmetry
    return L, d, d_inv_sqrt


def spectral_report(L, d):
    # numerical checks for Q2; eigh used only for verification
    evals, evecs = np.linalg.eigh(L)

    v = np.sqrt(d)
    v /= np.linalg.norm(v)  # trivial eigenvector D^{1/2} 1
    Lv = L @ v
    near_zero = int(np.sum(evals < 1e-6))  # ~ number of components
    k = max(near_zero, 1)

    return {
        "min_eig": float(evals.min()),
        "max_eig": float(evals.max()),
        "in_0_2": bool(evals.min() >= -1e-8 and evals.max() <= 2 + 1e-8),
        "trivial_residual": float(np.linalg.norm(Lv)),
        "trivial_rayleigh": float(v @ Lv),
        "num_near_zero": near_zero,
        "trivial_in_bottom_subspace": float(np.linalg.norm(evecs[:, :k] @ (evecs[:, :k].T @ v))),
        "evals": evals,
        "evecs": evecs,
    }


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_image import prepare
    from step2_weights import build_weight_matrix

    img, X, F, shape = prepare(size=50)
    W = build_weight_matrix(X, F, sigma_I=0.1, sigma_X=4.0, r=5.0)
    L, d, d_inv_sqrt = build_normalized_laplacian(W)
    rep = spectral_report(L, d)

    print(f"L_sym shape : {L.shape}, ||L - L^T||_F = {np.linalg.norm(L - L.T):.3e}")
    print(f"eig range = [{rep['min_eig']:.6e}, {rep['max_eig']:.6e}], all in [0, 2]? {rep['in_0_2']}")
    print(f"||L * (D^1/2 1)|| = {rep['trivial_residual']:.3e}, Rayleigh = {rep['trivial_rayleigh']:.3e}")
    print(f"num near-zero eig = {rep['num_near_zero']}")
    print(f"trivial in bottom subspace = {rep['trivial_in_bottom_subspace']:.6f}")

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(5, 3.2))
    plt.plot(np.arange(1, 21), rep["evals"][:20], "o-")
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.title("smallest 20 eigenvalues of L_sym")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "step3_eigs.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step3_eigs.png')}")
