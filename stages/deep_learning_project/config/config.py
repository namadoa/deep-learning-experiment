from dataclasses import dataclass
import typing as ty

@dataclass
class DataSetClass:
    dataset: str

@dataclass
class DataConfig:
    syndrome_data: DataSetClass
    non_syndrome_data: DataSetClass


@dataclass
class StagingConfig:
    """
    Arguments:
        data_processed_dataset: path processed dataset
        latitude_column: latitude column name (partner dataset)
        longitude_column: longitude column name (partner dataset)
    """
    syndrome_data: DataSetClass
    non_syndrome_data: DataSetClass


@dataclass
class SplittingDataConfig:
    """
    Arguments:
        test_val_size: proportion of test and val size E.g. [0.2, 0.1]
        root: site to save the Train/Val partioning datasets
        shuffle: Bool to indicate randomness (True) or not (False)
        unnecesary_columns: List of unnecesary columns for the modelling process
    """

    shuffle: bool 
    test_size: float
    training_data: DataSetClass
    testing_data: DataSetClass


@dataclass
class WandBConfig:
    tags: ty.List[str]
    parameters: ty.Dict[ty.Any, ty.Any]
    entity_name: str 
    project_name: str
    

@dataclass
class StaticParametersConfig:
    epochs: int
    batch_size: int


@dataclass
class ModuleConfig:
    name: str
    module: str


@dataclass
class BayesianOptimizerConfig:
    init_points: int
    n_inter: int


@dataclass
class TrainConfig:
    """
    Arguments:
        wandb_config: Weights and Bias configuration (it is associated
        with the model's hyperparameters)
        static_parameters: Dictionary with the static hyperparameters
        in this specific case (epochs, batch_size and random_state)
        optimizer: Optimizer module for the model
        preprocessing: Preprocessing module for the model
        scoring: Dictionary with the performance metrics.
        project_config: Weights and Bias configuration project
        models_dir: Directory to track genereated models.
    """

    wandb_config: WandBConfig
    static_parameters: StaticParametersConfig
    optimizer: ModuleConfig
    preprocessing: ModuleConfig
    bayesian_optimization: BayesianOptimizerConfig
    num_classes: int
    rotation_range: int
    num_augmentations: int
    random_state: int
    models_dir: str


@dataclass
class Config:
    """
    Arguments:
        data: data configuration
        model: model configuration
        training: training configuration
    """

    data: DataConfig
    staging: StagingConfig
    splitting_data: SplittingDataConfig
    modelling: TrainConfig
