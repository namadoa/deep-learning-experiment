import os
import numpy as np
import cv2
import h5py

from typing import Optional, Tuple, List

def load_images_from_folder(folder: str, target_size: Optional[Tuple[int, int]] = (512, 512)) -> List[np.ndarray]:
    """Load images from an specific folder

    Args:
        folder (str): Folder where the raw images are allocated
        target_size (Optional[Tuple[int, int]], optional): New size of the images. Defaults to (512, 512).

    Returns:
        List[np.ndarray]: List of resized images
    """
    resized_images = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        raw_image = cv2.imread(image_path)
        if raw_image is not None:
            resized_image = cv2.resize(raw_image)
        resized_images.append(resized_image)
    return resized_images

def normalize_images(images: List[np.ndarray]) -> np.ndarray:
    """Normalize images

    Args:
        images (List[np.ndarray]): List of images to be normalized

    Returns:
        np.ndarray: Array of normalized images
    """
    processed_images: List[np.ndarray] = [image / 255.0 for image in images]
    return np.array(processed_images)
