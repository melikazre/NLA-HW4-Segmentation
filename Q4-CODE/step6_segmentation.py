"""Step 6 (Q6): cluster the spectral embedding with k-means and visualize the segmentation."""

import numpy as np


def _kmeans_plusplus_init(Xf, k, rng):
    # k-means++ seeding for a more stable result
    n = Xf.shape[0]
    centers = [Xf[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            ((Xf[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(Xf[rng.choice(n, p=probs)])
    return np.array(centers, dtype=float)


def kmeans(Xf, k, n_iter=100, seed=0):
    # Lloyd's algorithm, implemented from scratch
    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus_init(Xf, k, rng)
    labels = np.full(Xf.shape[0], -1)
    for _ in range(n_iter):
        d2 = ((Xf[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = Xf[mask].mean(axis=0)
    return labels


def segment(L, d, k, mu=1e-3):
    """Return per-pixel labels for k clusters using the smallest nonzero modes."""
    from step4_deflation import compute_segmentation_eigenvectors

    _, Z = compute_segmentation_eigenvectors(L, d, k=k, mu=mu)
    Y = Z / np.sqrt(d)[:, None]  # back to the Ncut (random-walk) embedding y = D^{-1/2} z
    return kmeans(Y, k)


def segment_from_embedding(Z, d, k):
    """Same as segment() but reuses an already-computed embedding Z = [z2, ..., z_kmax]."""
    Y = (Z[:, : k - 1]) / np.sqrt(d)[:, None]
    return kmeans(Y, k)


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from step1_image import prepare
    from step2_weights import build_weight_matrix
    from step3_laplacian import build_normalized_laplacian
    from step4_deflation import compute_segmentation_eigenvectors

    img, X, F, shape = prepare(size=50)
    W = build_weight_matrix(X, F, sigma_I=0.1, sigma_X=4.0, r=5.0)
    L, d, _ = build_normalized_laplacian(W)

    # compute z2, z3, z4 once, then reuse subsets for k = 2, 3, 4
    ks = [2, 3, 4]
    _, Z = compute_segmentation_eigenvectors(L, d, k=max(ks), mu=1e-3)

    fig, axes = plt.subplots(1, 1 + len(ks), figsize=(3.2 * (1 + len(ks)), 3.4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], ks):
        labels = segment_from_embedding(Z, d, k)
        ax.imshow(labels.reshape(shape), cmap="tab10")
        ax.set_title(f"segmentation k={k}")
        ax.axis("off")
        counts = np.bincount(labels, minlength=k)
        print(f"k={k}: cluster sizes = {counts.tolist()}")

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "step6_segmentation.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step6_segmentation.png')}")
