import hydra
import logging
import numpy as np

from hydra.core.config_store import ConfigStore
from deep_learning_project.config import Config
from deep_learning_project.utils.image_preprocessing_utils import load_images_from_folder, normalize_images
from deep_learning_project.utils.h5_data_utils import save_to_h5

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path='../', config_name='params')
def process_dataset(cfg: Config) -> None:
    """Process image dataset

    Args:
        cfg (Config): project configuration
    """

    # loading images dataset
    syndrome_images = load_images_from_folder(cfg.data.syndrome_data.dataset)
    non_syndrome_images = load_images_from_folder(cfg.data.non_syndrome_data.dataset)
    logging.info("Images loaded succesfully")
    
    # normalize images dataset
    normalized_syndrome_images = normalize_images(syndrome_images)
    normalized_non_syndrome_images = normalize_images(non_syndrome_images)
    logging.info("Images normalized succesfully")

    # Create labels for the images
    syndrome_labels = np.ones(len(normalized_syndrome_images))
    non_syndrome_labels = np.zeros(len(normalized_non_syndrome_images))
    # saving pre-processed images to HDF5 files
    save_to_h5(normalized_syndrome_images, syndrome_labels, filename=cfg.staging.syndrome_data.dataset)
    save_to_h5(normalized_non_syndrome_images, non_syndrome_labels, filename=cfg.staging.non_syndrome_data.dataset)
    logging.info(f"Normalized images correctly saved in H5 format")

if __name__ == "__main__":
    process_dataset()
