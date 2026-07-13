"""Run every step in order and (re)generate all output figures.

Usage (from anywhere):
    python3 run_all.py
or:
    python3 code/run_all.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "step1_image.py",      # prepare image + features
    "step2_weights.py",    # similarity matrix W (Q1)
    "step3_laplacian.py",  # normalized Laplacian (Q2)
    "step5_plu.py",        # PLU decomposition self-test (Q5)
    "step6_segmentation.py",  # k-means segmentation k=2,3,4 (Q6)
    "step7_lanczos.py",    # Lanczos on a large sparse image (Q7)
    "step8_zebra.py",      # zebra demo (optional comparison)
]

if __name__ == "__main__":
    sys.path.insert(0, HERE)  # let steps import each other regardless of cwd
    for step in STEPS:
        print(f"\n{'=' * 60}\n  running {step}\n{'=' * 60}")
        runpy.run_path(os.path.join(HERE, step), run_name="__main__")
    print("\nDone. All figures are in code/outputs/.")
