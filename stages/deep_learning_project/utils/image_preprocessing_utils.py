import os
import numpy as np
import cv2
import h5py
import logging

from typing import Optional, Tuple, List
from deep_learning_project.errors import ResizeImageError

def load_images_from_folder(folder: str, target_size: Optional[Tuple[int, int]] = (512, 512)) -> List[np.ndarray]:
    """Load images from a specific folder and resize them.

    Args:
        folder (str): Folder where the raw images are located.
        target_size (Optional[Tuple[int, int]], optional): New size of the images. Defaults to (512, 512).

    Returns:
        List[np.ndarray]: List of resized images.
    """
    resized_images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            raw_image = cv2.imread(image_path)
            if raw_image is not None:
                resized_image = cv2.resize(raw_image, target_size)
                resized_images.append(resized_image)
            else:
                logging.error("The image can't be resized", extra={"raw_image": raw_image})
                raise ResizeImageError("The image can't be resized")
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
