from PIL import Image, ImageFilter
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
    #print(magnitude)
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

    return {
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "radial_avg1": radial_avg1_trimmed,
        "radial_avg2": radial_avg2_trimmed,
        "common_length": common_length,
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
   """  image_path = r"C:\\swetha\\sample_21_crop.jpg"
    image = Image.open(image_path)
    normalized = normalize_illumination(image)
    normalized.save(r"C:\\swetha\\normalized_sample_21.jpg")
    print("Saved normalized image to normalized_sample_21.jpg")
    [radii, radial_avg] = computeradialavg(normalized)
    
    image_path2 = r"C:\\swetha\\sample_22_crop.jpg"
    image2 = Image.open(image_path2)
    normalized2 = normalize_illumination(image2)
    normalized2.save(r"C:\\swetha\\normalized_sample_22.jpg")
    print("Saved normalized image to normalized_sample_22.jpg")
    [radii2, radial_avg2] = computeradialavg(normalized2)
    
    similaitymeasure = similaitymeas(radial_avg, radial_avg2)
    print(f"Cosine Similarity: {similaitymeasure:.4f}")
    euclidean_radial_distance = euclidean_radial_distance(radial_avg, radial_avg2)
    print(f"Euclidean Distance: {euclidean_radial_distance:.4f}")
    # Plot radial averages
    plot_radial_avg(radii, radial_avg, title="Radial Average - Sample 12", show=True)
    plot_radial_avg(radii2, radial_avg2, title="Radial Average - Sample 11", show=True)  """