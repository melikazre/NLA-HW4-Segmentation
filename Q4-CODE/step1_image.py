"""Step 1: load the image and build per-pixel features."""

import numpy as np


def make_synthetic_image(size=50):
    # bright disk on a dark background, values in [0, 1]
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.full((size, size), 0.15)
    img[dist <= size / 4] = 0.85
    noise = np.random.default_rng(0).normal(0, 0.02, img.shape)
    return np.clip(img + noise, 0, 1)


def load_image(path, size=50, grayscale=True):
    from PIL import Image
    im = Image.open(path).convert("L" if grayscale else "RGB")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=float) / 255.0


def build_features(img):
    # X: pixel coords (n, 2), F: intensities (n, c)
    H, W = img.shape[:2]
    F = img.reshape(H * W, -1)
    rows, cols = np.mgrid[0:H, 0:W]
    X = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(float)
    return X, F, (H, W)


def prepare(path=None, size=50, grayscale=True):
    img = make_synthetic_image(size) if path is None else load_image(path, size, grayscale)
    X, F, shape = build_features(img)
    return img, X, F, shape


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, X, F, shape = prepare(size=50)
    print(f"image {shape} -> n = {X.shape[0]} pixels, X {X.shape}, F {F.shape}")
    print(f"intensity range: [{F.min():.3f}, {F.max():.3f}]")

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.title(f"prepared image {shape[0]}x{shape[1]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "step1_image.png"), dpi=120)
    print(f"saved: {os.path.join(OUT, 'step1_image.png')}")
