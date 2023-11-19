import hydra
import numpy as np
import logging

from hydra.core.config_store import ConfigStore
from sklearn.model_selection import train_test_split
from deep_learning_project.config import Config
from deep_learning_project.utils.h5_data_utils import load_data_from_h5_in_batches, save_to_h5

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="../", config_name="params")
def split_data(cfg: Config) -> None:
    """Splitting image dataset

    Args:
        cfg (Config): project configuration
    """

    # Initialize empty lists for images and labels
    syndrome_images, syndrome_labels = [], []
    non_syndrome_images, non_syndrome_labels = [], []

    # Load and append syndrome data in batches
    for img_batch, lbl_batch in load_data_from_h5_in_batches(cfg.staging.syndrome_data.dataset, 10):
        syndrome_images.append(img_batch)
        syndrome_labels.append(lbl_batch)

    # Load and append non-syndrome data in batches
    for img_batch, lbl_batch in load_data_from_h5_in_batches(cfg.staging.non_syndrome_data.dataset, 10):
        non_syndrome_images.append(img_batch)
        non_syndrome_labels.append(lbl_batch)

    # Concatenate the data arrays
    X = np.concatenate(syndrome_images + non_syndrome_images, axis=0)
    y = np.concatenate(syndrome_labels + non_syndrome_labels)

    logging.info("Dataset concatenated successfully")

    # Splitting image dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.splitting_data.test_size, 
        random_state=42,
    )

    logging.info("Splitting image dataset into train/test")

    # Saving train/test image dataset
    save_to_h5(X_train, y_train, cfg.splitting_data.training_data.dataset)
    save_to_h5(X_test, y_test, cfg.splitting_data.testing_data.dataset)

    logging.info("Dataset split and correctly saved in H5 format")

if __name__ == "__main__":
    split_data()
