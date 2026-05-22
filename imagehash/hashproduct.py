from PIL import Image
import imagehash


def hash_image_file(image_path, hash_size=8):
    """Return an image hash for the given image file path.

    Args:
        image_path (str): Path to the image file.
        hash_size (int): Size of the perceptual hash (default 8).

    Returns:
        imagehash.ImageHash: The image hash object.
    """
    image = Image.open(image_path)
    return imagehash.whash(image, hash_size=hash_size)


def compute_similarity(hash1, hash2):
    """Compute similarity metrics between two image hashes.

    Args:
        hash1 (imagehash.ImageHash): First image hash.
        hash2 (imagehash.ImageHash): Second image hash.

    Returns:
        tuple[int, float]: Hamming distance and normalized similarity [0.0, 1.0].
    """
    distance = hash1 - hash2
    total_bits = hash1.hash.size
    similarity = 1.0 - (distance / total_bits)
    return distance, similarity


if __name__ == "__main__":
    #image_path = 'C:\swetha\sample_11_crop.jpg'
    image_path='C:\swetha\mfg-product-fingerprint\parts\sample_21.jpg'
    h1 = hash_image_file(image_path, hash_size=8)
    print("Hash 1:", str(h1))

    #image_path2 = 'C:\swetha\sample_12_crop.jpg'
    image_path2='C:\swetha\mfg-product-fingerprint\parts\sample_31.jpg'
    h2 = hash_image_file(image_path2, hash_size=8)
    print("Hash 2:", str(h2))

    distance, similarity = compute_similarity(h1, h2)
    print("Hamming distance:", distance)
    print(f"Similarity score: {similarity:.4f}")
    print(f"Similarity percentage: {similarity * 100:.1f}%")
