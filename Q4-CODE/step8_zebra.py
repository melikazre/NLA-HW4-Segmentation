"""Bonus: segment the real Shi & Malik zebra image with the same sparse Lanczos pipeline.

Real images only carry a single grayscale-intensity feature here, so the segmentation
captures broad scene regions (the contour over the zebra's back, foreground vs.
background) rather than a perfect object mask -- exactly the behaviour discussed in
the report. Everything (sparse Laplacian, Lanczos, k-means) is our own code.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from step7_lanczos import build_sparse_laplacian, smallest_eigenvectors_lanczos
from step6_segmentation import kmeans


def load_gray(path, width=120):
    im = Image.open(path).convert("L")
    w, h = im.size
    im = im.resize((width, round(h * width / w)), Image.BILINEAR)
    return np.asarray(im, dtype=float) / 255.0


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "outputs")
    img = load_gray(os.path.join(HERE, "zebra.png"), width=120)
    L, d = build_sparse_laplacian(img, sigma_I=0.1, sigma_X=6.0, r=7.0)
    n = L.shape[0]
    print(f"zebra image {img.shape} -> n = {n}, nnz(L) = {L.nnz} "
          f"({100 * L.nnz / n**2:.3f}% dense)")

    ks = [2, 3, 4]
    lam, V = smallest_eigenvectors_lanczos(L, n_vectors=max(ks) + 1, m=160)
    print("smallest eigenvalues (Lanczos):", np.array2string(lam, precision=6))

    fig, axes = plt.subplots(1, 1 + len(ks), figsize=(3.2 * (1 + len(ks)), 3.0))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("zebra (grayscale)")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], ks):
        Y = V[:, 1:k] / np.sqrt(d)[:, None]
        labels = kmeans(Y, k)
        ax.imshow(labels.reshape(img.shape), cmap="tab10")
        ax.set_title(f"segmentation k={k}")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "step8_zebra.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step8_zebra.png')}")
