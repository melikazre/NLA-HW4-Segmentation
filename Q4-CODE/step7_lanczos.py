"""Step 7 (Q7): Lanczos algorithm (from scratch) for segmenting a large, sparse image.

Lanczos only touches the matrix through matrix-vector products, so we keep W (and
hence L_sym) sparse and never form/factor/invert it. To reach the *smallest*
eigenvalues of L_sym (the ones useful for segmentation) we run Lanczos on
M = 2I - L_sym, whose *largest* eigenvalues correspond to the smallest of L_sym.
"""

import numpy as np
import scipy.sparse as sp


def make_large_image(size=120):
    # 2x2 quadrants with a diagonal intensity gradient: four balanced, connected
    # regions with mild contrast -> well-separated small eigenvalues, ideal for
    # single-vector Lanczos (no clustered near-zero eigenvalues).
    img = np.zeros((size, size))
    h = size // 2
    img[:h, :h] = 0.20  # top-left
    img[:h, h:] = 0.40  # top-right
    img[h:, :h] = 0.60  # bottom-left
    img[h:, h:] = 0.80  # bottom-right
    noise = np.random.default_rng(0).normal(0, 0.02, img.shape)
    return np.clip(img + noise, 0, 1)


def build_sparse_laplacian(img, sigma_I=0.1, sigma_X=4.0, r=5.0):
    # build a sparse W exploiting the grid: only pixels within radius r interact
    H, W_ = img.shape
    n = H * W_
    rad = int(np.floor(r))
    idx = np.arange(n).reshape(H, W_)

    rows, cols, vals = [], [], []
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            d2 = dy * dy + dx * dx
            if d2 == 0 or d2 >= r * r:
                continue
            ys = slice(max(0, -dy), H - max(0, dy))
            xs = slice(max(0, -dx), W_ - max(0, dx))
            yt = slice(max(0, dy), H - max(0, -dy))
            xt = slice(max(0, dx), W_ - max(0, -dx))
            src = idx[ys, xs].ravel()
            dst = idx[yt, xt].ravel()
            dF2 = (img[ys, xs].ravel() - img[yt, xt].ravel()) ** 2
            w = np.exp(-dF2 / sigma_I ** 2) * np.exp(-d2 / sigma_X ** 2)
            rows.append(src)
            cols.append(dst)
            vals.append(w)

    W = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    W = 0.5 * (W + W.T)  # enforce exact symmetry

    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    Dis = sp.diags(d_inv_sqrt)
    L = sp.identity(n, format="csr") - (Dis @ W @ Dis)
    return L.tocsr(), d


def lanczos(matvec, n, m, seed=0):
    """Lanczos on a symmetric operator given by matvec; full reorthogonalization.

    Returns (alpha, beta, Q) with the tridiagonal entries and the Krylov basis.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, m + 1))
    alpha = np.zeros(m)
    beta = np.zeros(m + 1)

    q = rng.standard_normal(n)
    Q[:, 0] = q / np.linalg.norm(q)

    for j in range(m):
        w = matvec(Q[:, j])
        if j > 0:
            w = w - beta[j] * Q[:, j - 1]
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j]
        # full reorthogonalization to fight loss of orthogonality in floating point
        w = w - Q[:, : j + 1] @ (Q[:, : j + 1].T @ w)
        beta[j + 1] = np.linalg.norm(w)
        if beta[j + 1] < 1e-12:
            return alpha[: j + 1], beta[1: j + 1], Q[:, : j + 1]
        Q[:, j + 1] = w / beta[j + 1]

    return alpha, beta[1:m], Q[:, :m]


def smallest_eigenvectors_lanczos(L, n_vectors, m=80, seed=0):
    """Smallest eigenpairs of sparse symmetric L via Lanczos on M = 2I - L."""
    matvec = lambda v: 2.0 * v - L @ v  # M = 2I - L  (sparse matvec only)
    alpha, beta, Q = lanczos(matvec, L.shape[0], m, seed=seed)

    # eigen-decomposition of the small tridiagonal T_m (m is tiny -> cheap)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    theta, S = np.linalg.eigh(T)  # Ritz values/vectors of M

    lam = 2.0 - theta  # back to eigenvalues of L
    order = np.argsort(lam)  # ascending: smallest eigenvalues of L first
    lam = lam[order]
    V = Q @ S[:, order]  # Ritz vectors approximate eigenvectors of L
    return lam[:n_vectors], V[:, :n_vectors]


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from step6_segmentation import kmeans

    size = 120
    img = make_large_image(size)
    # sigma_I = 0.2 keeps the four quadrants connected with a clear spectral gap,
    # so single-vector Lanczos converges cleanly to each small eigenvalue.
    L, d = build_sparse_laplacian(img, sigma_I=0.2, sigma_X=4.0, r=5.0)
    n = L.shape[0]
    print(f"large image {img.shape} -> n = {n}, nnz(L) = {L.nnz} "
          f"({100 * L.nnz / n**2:.3f}% dense)")

    # smallest few modes: index 0 is the trivial one, z2.. are the next ones
    k = 4
    lam, V = smallest_eigenvectors_lanczos(L, n_vectors=k + 1, m=120)
    print("smallest eigenvalues (Lanczos):", np.array2string(lam, precision=6))

    Z = V[:, 1:k]               # drop the trivial mode, keep z2..zk
    Y = Z / np.sqrt(d)[:, None]  # Ncut embedding
    labels = kmeans(Y, k)
    print(f"k={k}: cluster sizes = {np.bincount(labels, minlength=k).tolist()}")

    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(f"large image {size}x{size}")
    ax[0].axis("off")
    ax[1].imshow(labels.reshape(img.shape), cmap="tab10")
    ax[1].set_title(f"Lanczos segmentation k={k}")
    ax[1].axis("off")
    plt.tight_layout()

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "step7_lanczos.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step7_lanczos.png')}")
