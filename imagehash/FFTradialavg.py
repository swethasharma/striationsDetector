from PIL import Image, ImageFilter
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_illumination(image, blur_radius=15, target_mean=128):
    """Normalize illumination in an image.

    This function estimates the local illumination field using a Gaussian blur
    and removes it from the input image to reduce lighting variation.

    Args:
        image (PIL.Image.Image): Source image.
        blur_radius (int): Gaussian blur radius used to estimate illumination.
        target_mean (int): Mean brightness target after normalization.

    Returns:
        PIL.Image.Image: Illumination-normalized image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")

    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    arr = np.asarray(image).astype(np.float32)
    if image.mode == "L":
        arr = arr[..., np.newaxis]

    normalized_channels = []
    for c in range(arr.shape[2]):
        channel = arr[..., c]
        blurred = np.asarray(Image.fromarray(channel.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius))).astype(np.float32)
        result = channel - blurred + target_mean
        result = np.clip(result, 0, 255)
        normalized_channels.append(result.astype(np.uint8))

    normalized = np.stack(normalized_channels, axis=-1)
    if image.mode == "L":
        normalized = normalized[..., 0]

    return Image.fromarray(normalized, mode=image.mode)


def computeradialavg(image, grayscale=True):
    """Compute the radial average of the FFT magnitude spectrum.

    Args:
        image (PIL.Image.Image | np.ndarray): Input image.
        grayscale (bool): If True, convert the image to grayscale before FFT.

    Returns:
        tuple[np.ndarray, np.ndarray]: (radii, radial_average)
            radii: integer radial distances from the FFT center.
            radial_average: mean magnitude values for each radius.
    """
    if isinstance(image, Image.Image):
        if grayscale:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
        else:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
    else:
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)

    fft = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    height, width = magnitude.shape
    y, x = np.indices((height, width))
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radii = np.hypot(x - center_x, y - center_y).astype(np.int32)

    radial_sum = np.bincount(radii.ravel(), weights=magnitude.ravel())
    radial_count = np.bincount(radii.ravel())
    radial_average = radial_sum / radial_count

    return np.arange(radial_average.size), radial_average


def similaitymeas(radial_avg1, radial_avg2):
    """Calculate cosine similarity between two radial average vectors.

    Args:
        radial_avg1 (np.ndarray): Radial average of the first image.
        radial_avg2 (np.ndarray): Radial average of the second image.

    Returns:
        float: Cosine similarity score in the range [-1, 1].
    """
    radial_avg1 = np.asarray(radial_avg1, dtype=np.float32).ravel()
    radial_avg2 = np.asarray(radial_avg2, dtype=np.float32).ravel()

    if len(radial_avg1) != len(radial_avg2):
        raise ValueError("Radial averages must have the same length")

    dot_product = np.dot(radial_avg1, radial_avg2)
    norm1 = np.linalg.norm(radial_avg1)
    norm2 = np.linalg.norm(radial_avg2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_similarity = dot_product / (norm1 * norm2)
    return float(cosine_similarity)


def euclidean_radial_distance(radial_avg1, radial_avg2):
    """Calculate Euclidean distance between two radial average vectors.

    Args:
        radial_avg1 (np.ndarray): Radial average of the first image.
        radial_avg2 (np.ndarray): Radial average of the second image.

    Returns:
        float: Euclidean distance between the two radial averages.
    """
    radial_avg1 = np.asarray(radial_avg1, dtype=np.float32).ravel()
    radial_avg2 = np.asarray(radial_avg2, dtype=np.float32).ravel()

    if len(radial_avg1) != len(radial_avg2):
        raise ValueError("Radial averages must have the same length")

    diff = radial_avg1 - radial_avg2
    return float(np.linalg.norm(diff))


def correlation_radial(radial_avg1, radial_avg2, normalize=True):
    """Compute cross-correlation between two radial average vectors.

    Args:
        radial_avg1 (array-like): First radial average vector.
        radial_avg2 (array-like): Second radial average vector.
        normalize (bool): If True, normalize by product of norms.

    Returns:
        lags (np.ndarray): Array of integer lags (negative means second leads).
        corr (np.ndarray): Cross-correlation values (length = N1+N2-1).
        lag_of_max (int): Lag at which correlation is maximal.
        max_val (float): Maximum correlation value.
    """
    a = np.asarray(radial_avg1, dtype=np.float64).ravel()
    b = np.asarray(radial_avg2, dtype=np.float64).ravel()

    if a.size == 0 or b.size == 0:
        raise ValueError("Input radial averages must be non-empty")

    a_zm = a - a.mean()
    b_zm = b - b.mean()

    corr = np.correlate(a_zm, b_zm, mode="full")

    if normalize:
        denom = np.linalg.norm(a_zm) * np.linalg.norm(b_zm)
        if denom != 0:
            corr = corr / denom

    lags = np.arange(-b.size + 1, a.size)
    max_idx = int(np.argmax(corr))
    lag_of_max = int(lags[max_idx])
    max_val = float(corr[max_idx])

    return lags, corr, lag_of_max, max_val


def _normalize_plot_vectors(x, y, shape):
    """Normalize plot coordinates into pixel indices for a fixed image size."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Plot vectors must be non-empty")

    if x.size != y.size:
        raise ValueError("Radii and radial average vectors must have the same length")

    x = x - x.min()
    if x.max() != 0:
        x = x / x.max()

    y = y - y.min()
    if y.max() != 0:
        y = y / y.max()

    width = shape[1]
    height = shape[0]
    xs = np.round(x * (width - 1)).astype(np.int32)
    ys = np.round((1.0 - y) * (height - 1)).astype(np.int32)
    return xs, ys


def _plot_line_image(xs, ys, shape):
    """Rasterize a 1D plot into a 2D image by drawing a polyline."""
    image = np.zeros(shape, dtype=np.float32)
    for start, end in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy, 1)
        for t in range(steps + 1):
            xi = int(round(x0 + (x1 - x0) * t / steps))
            yi = int(round(y0 + (y1 - y0) * t / steps))
            image[yi, xi] = 1.0
    return image


def _correlate2d(a, b):
    """Compute full 2D cross-correlation using FFT."""
    shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    fa = np.fft.fft2(a, shape)
    fb = np.fft.fft2(b, shape)
    corr = np.fft.ifft2(fa * np.conj(fb))
    return np.real(corr)


def correlation_2d_plots(radii1, radial_avg1, radii2, radial_avg2, image_size=(512, 512), normalize=True):
    """Compute 2D cross-correlation between the plot images of two radial average vectors.

    Args:
        radii1 (array-like): Radii for the first plot.
        radial_avg1 (array-like): Radial average values for the first plot.
        radii2 (array-like): Radii for the second plot.
        radial_avg2 (array-like): Radial average values for the second plot.
        image_size (tuple[int, int]): Output 2D image size for rasterizing the plots.
        normalize (bool): If True, normalize the correlation by vector norms.

    Returns:
        tuple:
            corr (np.ndarray): Full 2D cross-correlation map.
            max_position (tuple[int, int]): Position of the maximum correlation.
            max_value (float): Maximum correlation value.
            plot_image1 (np.ndarray): Rasterized plot image for the first radial average.
            plot_image2 (np.ndarray): Rasterized plot image for the second radial average.
    """
    xs1, ys1 = _normalize_plot_vectors(radii1, radial_avg1, image_size)
    xs2, ys2 = _normalize_plot_vectors(radii2, radial_avg2, image_size)

    plot_image1 = _plot_line_image(xs1, ys1, image_size)
    plot_image2 = _plot_line_image(xs2, ys2, image_size)

    corr = _correlate2d(plot_image1, plot_image2)
    if normalize:
        denom = np.linalg.norm(plot_image1) * np.linalg.norm(plot_image2)
        if denom != 0:
            corr = corr / denom

    max_idx = int(np.argmax(corr))
    max_position = np.unravel_index(max_idx, corr.shape)
    max_value = float(corr.ravel()[max_idx])

    return corr, max_position, max_value, plot_image1, plot_image2


def getsimilarityscore(image1, image2, grayscale=True):
    """Compute similarity scores between two images.

    This function normalizes illumination, computes radial averages from the FFT
    magnitude spectrum, and returns cosine similarity plus Euclidean distance.

    Args:
        image1 (PIL.Image.Image | np.ndarray | str): First image or image path.
        image2 (PIL.Image.Image | np.ndarray | str): Second image or image path.
        grayscale (bool): If True, convert images to grayscale before FFT.

    Returns:
        dict: {
            "cosine_similarity": float,
            "euclidean_distance": float,
            "radial_avg1": np.ndarray,
            "radial_avg2": np.ndarray,
            "common_length": int,
        }
    """
    def _load_image(image):
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8))
        raise TypeError("image1 and image2 must be PIL.Image.Image, numpy.ndarray, or file path string")

    img1 = _load_image(image1)
    img2 = _load_image(image2)

    norm1 = normalize_illumination(img1)
    norm2 = normalize_illumination(img2)

    radii1, radial_avg1 = computeradialavg(norm1, grayscale=grayscale)
    radii2, radial_avg2 = computeradialavg(norm2, grayscale=grayscale)

    common_length = min(len(radial_avg1), len(radial_avg2))
    if common_length == 0:
        raise ValueError("Computed radial averages are empty")

    radial_avg1_trimmed = radial_avg1[:common_length]
    radial_avg2_trimmed = radial_avg2[:common_length]
    cosine_similarity = similaitymeas(radial_avg1_trimmed, radial_avg2_trimmed)
    euclidean_distance = euclidean_radial_distance(radial_avg1_trimmed, radial_avg2_trimmed)

    # compute cross-correlation of the radial averages
    try:
        lags, corr_vals, corr_lag_of_max, corr_max_val = correlation_radial(radial_avg1_trimmed, radial_avg2_trimmed, normalize=True)
    except Exception:
        lags, corr_vals, corr_lag_of_max, corr_max_val = None, None, None, 0.0

    return {
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "radial_avg1": radial_avg1_trimmed,
        "radial_avg2": radial_avg2_trimmed,
        "common_length": common_length,
        "corr_lag_of_max": corr_lag_of_max,
        "corr_max_value": corr_max_val,
    }


def plot_radial_avg(radii, radial_avg, title="Radial Average Spectrum", show=True):
    """Plot the radial average of the FFT magnitude spectrum.

    Args:
        radii (np.ndarray): Radial distances from the FFT center.
        radial_avg (np.ndarray): Mean magnitude values for each radius.
        title (str): Title of the plot.
        show (bool): If True, display the plot; otherwise just return the figure.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(radii, radial_avg, linewidth=2, color="blue")
    ax.set_xlabel("Radial Distance (pixels)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale for better visualization

    if show:
        plt.tight_layout()
        plt.show()

    return fig


if __name__ == "__main__":
    def main():
        base_dir = Path(r"C:\swetha")
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

        image_paths = sorted(base_dir.glob("*_crop.jpg"))
        if not image_paths:
            print(f"No images found matching '*_crop.jpg' in {base_dir}")
            return

        print(f"Found {len(image_paths)} crop images in {base_dir}")

        normalized_cache = {}
        radial_cache = {}
        normalized_paths = {}
        plot_paths = {}

        for image_path in image_paths:
            image = Image.open(image_path)
            normalized = normalize_illumination(image)

            normalized_name = image_path.with_name(f"{image_path.stem}_normalized{image_path.suffix}")
            normalized.save(normalized_name)
            normalized_paths[image_path] = normalized_name
            print(f"Saved normalized image: {normalized_name}")

            radii, radial_avg = computeradialavg(normalized)
            radial_cache[image_path] = (radii, radial_avg)
            normalized_cache[image_path] = normalized

            plot_name = image_path.with_name(f"{image_path.stem}_radialavg.png")
            fig = plot_radial_avg(radii, radial_avg, title=f"Radial Average - {image_path.name}", show=False)
            fig.savefig(plot_name, dpi=200)
            plt.close(fig)
            plot_paths[image_path] = plot_name
            print(f"Saved radial average plot: {plot_name}")

        csv_path = base_dir / "similarity_results.csv"
        csv_rows = []

        print("\nComputing similarity scores for all image pairs:\n")
        for i, image_path1 in enumerate(image_paths):
            for image_path2 in image_paths[i + 1:]:
                score = getsimilarityscore(normalized_cache[image_path1], normalized_cache[image_path2])
                radii1, radial_avg1 = radial_cache[image_path1]
                radii2, radial_avg2 = radial_cache[image_path2]
                _, _, plot_corr_value, _, _ = correlation_2d_plots(
                    radii1, radial_avg1, radii2, radial_avg2, image_size=(512, 512), normalize=True
                )
                row = {
                    "image1": image_path1.name,
                    "image2": image_path2.name,
                    "plot_correlation": f"{plot_corr_value:.6f}",
                }
                csv_rows.append(row)
                print(f"{image_path1.name} vs {image_path2.name}: plot_corr={plot_corr_value:.6f}")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "image1",
                "image2",
                "plot_correlation",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Saved similarity results CSV: {csv_path}")

    main()
