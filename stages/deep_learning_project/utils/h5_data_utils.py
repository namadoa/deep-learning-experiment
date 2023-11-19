import h5py
import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Generator


def save_to_h5(images: np.ndarray, labels: Optional[np.ndarray] = None, filename: str = 'data.h5') -> None:
    """Save images (and optionally labels) in H5 format.

    Args:
        images (np.ndarray): Array of images to be stored.
        labels (Optional[np.ndarray]): Optional array of labels for the images.
        filename (str): Name of the H5 file.
    """
    with h5py.File(filename, 'w') as hf:
        logging.info("creating dataset")
        hf.create_dataset('images', data=images)
        if labels is not None:
            logging.info("creating dataset labels")
            hf.create_dataset('labels', data=labels)
        logging.info(f'saved dataset here {filename}')

def load_data_from_h5_in_batches(h5_path: str, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Load images and labels from H5 format in batches.

    Args:
        h5_path (str): Path to the H5 file.
        batch_size (int): Size of each batch.

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: Generator of tuple of images and labels arrays.
    """
    with h5py.File(h5_path, 'r') as hf:
        total_size = hf['images'].shape[0]
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            images = np.array(hf['images'][start_idx:end_idx])
            labels = np.array(hf['labels'][start_idx:end_idx]) if 'labels' in hf else None
            yield images, labels

def load_data_from_h5(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels from H5 format.

    Args:
        h5_path (str): Path to the H5 file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of images and labels arrays.
    """
    with h5py.File(h5_path, 'r') as hf:
        images = np.array(hf['images'])
        labels = np.array(hf['labels']) if 'labels' in hf else None
    return images, labels

def generate_tensor_from_h5(h5_path: str) -> tf.Tensor:
    """generate tensor from h5 format

    Args:
        h5_path (str): H5 path

    Returns:
        tf.Tensor: tensor of images
    """
    images = load_data_from_h5(h5_path)
    tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    return tensor
