"""normalize_image.py

Small CLI wrapper that uses normalize_illumination from FFTdialavg.py
to normalize illumination of a given image and save the output.

usage:
    python normalize_image.py [--root ROOT_PATH]

If --root is omitted the script will scan C:\\swetha for files matching `*_crop.jpg`.
"""
import argparse
from pathlib import Path
import csv
import itertools
from PIL import Image
import numpy as np

# import normalize function from existing module
from FFTradialavg import normalize_illumination


def compute_lbp(block):
    """Compute basic 8-bit local binary pattern for a 2D block."""
    if block.ndim != 2:
        raise ValueError("block must be a 2D array")

    padded = np.pad(block, pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
        padded[1:-1, 2:], padded[2:, 2:], padded[2:, 1:-1],
        padded[2:, 0:-2], padded[1:-1, 0:-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neigh in enumerate(neighbors):
        lbp |= ((neigh >= center).astype(np.uint8) << bit)
    return lbp


def _pad_to_block_multiple(gray, block_size=(64, 64)):
    """Pad 2D grayscale array on bottom/right so its shape is a multiple of block_size."""
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D array")
    bh, bw = block_size
    h, w = gray.shape
    pad_h = (-h) % bh
    pad_w = (-w) % bw
    if pad_h or pad_w:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
    return gray


def _pad_to_target_size(gray, target_size):
    """Pad 2D grayscale array on bottom/right to match target_size."""
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D array")
    target_h, target_w = target_size
    h, w = gray.shape
    if h > target_h or w > target_w:
        raise ValueError("Target size must be at least the image size")
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h or pad_w:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
    return gray


def block_lbp_features(image, block_size=(64, 64), target_size=None):
    """Compute block-wise LBP feature vector for an image.

    The image is divided into non-overlapping blocks of size 64x64. Each block
    produces a normalized 256-bin LBP histogram, and the final output is the
    concatenated feature vector for all blocks.
    """
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        gray = image.astype(np.uint8)
        if gray.ndim == 3:
            gray = np.asarray(Image.fromarray(gray).convert("L"), dtype=np.uint8)
    else:
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    if target_size is not None:
        gray = _pad_to_target_size(gray, target_size)
    else:
        gray = _pad_to_block_multiple(gray, block_size)
    bh, bw = block_size
    h, w = gray.shape
    features = []
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            block = gray[y:y + bh, x:x + bw]
            lbp = compute_lbp(block)
            hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            features.append(hist)

    if not features:
        raise ValueError("Image is smaller than one 64x64 block")

    return np.concatenate(features)


def glcm_properties(glcm):
    """Compute standard GLCM properties from a normalized co-occurrence matrix."""
    levels = glcm.shape[0]
    i = np.arange(levels).reshape((-1, 1))
    j = np.arange(levels).reshape((1, -1))

    contrast = np.sum(glcm * (i - j) ** 2)
    dissimilarity = np.sum(glcm * np.abs(i - j))
    homogeneity = np.sum(glcm / (1.0 + (i - j) ** 2))
    asm = np.sum(glcm ** 2)
    energy = np.sqrt(asm)

    mu_i = np.sum(i * glcm)
    mu_j = np.sum(j * glcm)
    sigma_i = np.sqrt(np.sum(((i - mu_i) ** 2) * glcm))
    sigma_j = np.sqrt(np.sum(((j - mu_j) ** 2) * glcm))
    if sigma_i == 0 or sigma_j == 0:
        correlation = 0.0
    else:
        correlation = np.sum(((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j))

    return np.array([contrast, dissimilarity, homogeneity, energy, asm, correlation], dtype=np.float32)


def compute_glcm(block, levels=8, distances=(1,), angles=(0,)):
    """Compute an average GLCM for the block across distances and angles."""
    if block.ndim != 2:
        raise ValueError("block must be a 2D array")

    bins = np.linspace(0, 256, levels + 1, endpoint=True)
    quantized = np.digitize(block, bins) - 1
    quantized = np.clip(quantized, 0, levels - 1)
    glcm_sum = np.zeros((levels, levels), dtype=np.float64)

    for d in distances:
        for angle in angles:
            dy = int(round(np.sin(angle) * d))
            dx = int(round(np.cos(angle) * d))
            if dy == 0 and dx == 0:
                continue
            padded = np.pad(quantized, pad_width=1, mode="edge")
            i = padded[1:-1, 1:-1]
            j = padded[1 + dy:1 + dy + quantized.shape[0], 1 + dx:1 + dx + quantized.shape[1]]
            coords = (i.ravel(), j.ravel())
            glcm = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(glcm, coords, 1)
            glcm_sum += glcm

    if glcm_sum.sum() == 0:
        return np.zeros((levels, levels), dtype=np.float32)
    return (glcm_sum / glcm_sum.sum()).astype(np.float32)


def block_glcm_features(image, block_size=(64, 64), levels=8, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), target_size=None):
    """Compute block-wise GLCM features for an image."""
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        gray = image.astype(np.uint8)
        if gray.ndim == 3:
            gray = np.asarray(Image.fromarray(gray).convert("L"), dtype=np.uint8)
    else:
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    if target_size is not None:
        gray = _pad_to_target_size(gray, target_size)
    else:
        gray = _pad_to_block_multiple(gray, block_size)
    bh, bw = block_size
    h, w = gray.shape
    features = []
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            block = gray[y:y + bh, x:x + bw]
            glcm = compute_glcm(block, levels=levels, distances=distances, angles=angles)
            features.append(glcm_properties(glcm))

    if not features:
        raise ValueError("Image is smaller than one 64x64 block")

    return np.concatenate(features)


def main():
    p = argparse.ArgumentParser(description="Scan a directory for *_crop.jpg images and compute block LBP/GLCM features")
    p.add_argument("--root", default=r"C:\\swetha", help="Root path to search for *_crop.jpg images")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    crop_images = sorted(root.glob("*_crop.jpg"))
    if not crop_images:
        raise SystemExit(f"No *_crop.jpg files found in {root}")

    max_w = 0
    max_h = 0
    for image_path in crop_images:
        with Image.open(image_path) as img:
            w, h = img.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    target_w = ((max_w + 63) // 64) * 64
    target_h = ((max_h + 63) // 64) * 64

    dict_lbp = {}
    dict_glcm = {}

    for image_path in crop_images:
        with Image.open(image_path) as img:
            dict_lbp[str(image_path)] = block_lbp_features(img, target_size=(target_h, target_w))
            dict_glcm[str(image_path)] = block_glcm_features(img, target_size=(target_h, target_w))

    print(f"Processed {len(crop_images)} crop images from {root}")
    print(f"Using common padded target size: {target_w}x{target_h}")
    print(f"LBP features stored in dict_lbp with {len(dict_lbp)} entries")
    print(f"GLCM features stored in dict_glcm with {len(dict_glcm)} entries")

    # compute pairwise distances and save CSV
    keys = list(dict_lbp.keys())
    csv_path = root / "lbp_glcm.csv"
    header = [
        "img1", "img2",
        "cosine_lbp", "euclidean_lbp",
        "cosine_glcm", "euclidean_glcm",
        "lbp_vec_1", "lbp_vec_2", "glcm_vec_1", "glcm_vec_2",
    ]

    def cosine_distance(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for k1, k2 in itertools.combinations(keys, 2):
            lbp1 = dict_lbp[k1]
            lbp2 = dict_lbp[k2]
            glcm1 = dict_glcm[k1]
            glcm2 = dict_glcm[k2]

            cos_lbp = cosine_distance(lbp1, lbp2)
            eu_lbp = float(np.linalg.norm(np.asarray(lbp1, dtype=np.float64) - np.asarray(lbp2, dtype=np.float64)))

            cos_glcm = cosine_distance(glcm1, glcm2)
            eu_glcm = float(np.linalg.norm(np.asarray(glcm1, dtype=np.float64) - np.asarray(glcm2, dtype=np.float64)))

            # store vector strings for reference
            lbp1_s = np.array2string(np.asarray(lbp1), separator=',')
            lbp2_s = np.array2string(np.asarray(lbp2), separator=',')
            glcm1_s = np.array2string(np.asarray(glcm1), separator=',')
            glcm2_s = np.array2string(np.asarray(glcm2), separator=',')

            writer.writerow([k1, k2, cos_lbp, eu_lbp, cos_glcm, eu_glcm, lbp1_s, lbp2_s, glcm1_s, glcm2_s])

    print(f"Wrote pairwise distances and vectors to: {csv_path}")


if __name__ == "__main__":
    main()
