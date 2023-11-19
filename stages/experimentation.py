import hydra
from hydra.core.config_store import ConfigStore
from deep_learning_project.config import Config
from deep_learning_project.model.trainer import Learner
from deep_learning_project.utils.h5_data_utils import load_data_from_h5

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="../", config_name="params")
def train(cfg: Config) -> None:
    """Experimentation process

    Args:
        cfg (Config): project configuration
    """
    # Loading training and testing dataset.
    X_train, y_train = load_data_from_h5(cfg.splitting_data.training_data.dataset)
    X_test, y_test = load_data_from_h5(cfg.splitting_data.testing_data.dataset)
    
    # Intialize the Learner and start training
    learner = Learner(X_train, y_train, X_test, y_test, cfg)
    learner.initialize_train()


if __name__ == "__main__":
    train()
