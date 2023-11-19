import h5py
import numpy as np
import tensorflow as tf

def save_to_h5(images: np.ndarray, labels: Optional[np.ndarray] = None, filename: str = 'data.h5') -> None:
    """Save images (and optionally labels) in H5 format.

    Args:
        images (np.ndarray): Array of images to be stored.
        labels (Optional[np.ndarray]): Optional array of labels for the images.
        filename (str): Name of the H5 file.
    """
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('images', data=images)
        if labels is not None:
            hf.create_dataset('labels', data=labels)

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
