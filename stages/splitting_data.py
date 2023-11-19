import hydra
from hydra.core.config_store import ConfigStore
from geo_project.config import Config
from geo_project.data import ParquetDataLoader
from geo_project.model.train_val_split import RandomStratifiedSplit

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="../", config_name="params")
def split_data(cfg: Config) -> None:
    """Splitting image dataset

    Args:
        cfg (Config): project configuration
    """

    # Loading processed dataset.
    syndrome_images, syndrome_labels = load_data_from_h5(cfg.staging.syndrome_data.dataset)
    non_syndrome_images, non_syndrome_labels = load_data_from_h5(cfg.staging.non_syndrome_data.dataset)

    # Concatenate the data arrays and create labels (0 for non-syndrome, 1 for syndrome)
    X = np.concatenate([non_syndrome_images, syndrome_images], axis=0)
    y = np.concatenate([non_syndrome_labels, syndrome_labels])
    
    # Splitting image dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.splitting_data.test_size, 
        stratify=y, random_state=cfg.modelling.random_state,
        shuffle=cfg.splitting_data.shuffle
    )

    # Saving train/test image dataset
    save_to_h5(X_train, y_train, cfg.splitting_data.training_data.dataset)
    save_to_h5(X_test, y_test, cfg.splitting_data.testing_data.dataset)


if __name__ == "__main__":
    split_data()
