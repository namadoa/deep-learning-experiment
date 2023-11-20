import wandb
import os
import keras
import random
import logging
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf

from numpy import ndarray
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from deep_learning_project.config import Config
from deep_learning_project.utils.tools import import_name


@dataclass
class BaseLearner(ABC):

    @abstractmethod
    def train_evaluate(self, conf: Dict) -> None:
        ...

    @abstractmethod
    def base_trainer(self, dropout: float, learning_rate: float) -> Sequential:
        ...


class Learner(BaseLearner):
    def __init__(
            self,
            X_train: ndarray = np.array([]),
            y_train: ndarray = np.array([]),
            X_test: ndarray = np.array([]),
            y_test: ndarray = np.array([]),
            config: Config = {}
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.input_shape = self.X_train.shape[1:]
        self.num_classes = self.config.modelling.num_classes
        self.optimizer = import_name(self.config.modelling.optimizer.module,
                                     self.config.modelling.optimizer.name)
    
    def augment_data(self, images: np.ndarray, num_augmentations: Optional[int] =1):
        """
        Augments the given images by applying Keras transformations for data augmentation.

        Parameters:
            images (np.ndarray): An array of images to be augmented.
            num_augmentations (Optional[int]): The number of augmentations to apply to each image. Defaults to 1.

        Returns:
            np.ndarray: An array of augmented images.
        """

        # Define Keras transformations for data augmentation
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=self.config.modelling.rotation_range
        )

        n, height, width, channels = images.shape

        # Initialize an array to store augmented images
        augmented_images = np.zeros((n * (num_augmentations + 1), height, width, channels), dtype=np.uint8)

        # Apply transformations and store augmented images
        for i in range(n):
            # Original image
            augmented_images[i * (num_augmentations + 1)] = images[i]

            for j in range(num_augmentations):
                # Apply transformations
                transformed_image = datagen.random_transform(images[i])

                # Store in the augmented array
                augmented_images[i * (num_augmentations + 1) + j + 1] = transformed_image

        return augmented_images
    
    def setup_tensorflow(self):
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(self.config.modelling.random_state)
        np.random.seed(self.config.modelling.random_state)
        tf.random.set_seed(self.config.modelling.random_state)

    def base_trainer(self, dropout: float = 0.2, learning_rate: float = 1e-4) -> tf.keras.Model:
        """
        Trains a base model for binary classification using different architectures.
        
        Returns:
        - model: A trained model for binary classification.
        """
        input_shape = self.X_train.shape[1:]
        num_classes = self.config.modelling.num_classes
        architecture = self.config.modelling.architecture

        if architecture == 'AlexNet':
            model = self._build_alexnet(input_shape, dropout)
        elif architecture == 'ResNet50':
            model = self._build_resnet50(input_shape, dropout)
        elif architecture == 'MobileNet':
            base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
            model = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])
        elif architecture == 'DenseNet':
            base_model = tf.keras.applications.DenseNet121(input_shape=input_shape, include_top=False, weights=None)
            model = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])
        elif architecture == 'Inception':
            base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
            model = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model.add(Dense(num_classes, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=self.optimizer(learning_rate=learning_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy', keras.metrics.AUC(curve='ROC', name='roc_auc')]
        )
        return model

    def _build_resnet50(self, input_shape: Tuple[int, int, int], dropout: float) -> Sequential:
        """
        Builds a ResNet50 model with the given input shape, number of classes, dropout rate, and learning rate.

        Parameters:
            input_shape: The shape of the input data.
            num_classes: The number of classes for classification.
            dropout: The dropout rate.
            learning_rate: The learning rate for the optimizer.

        Returns:
            The compiled ResNet50 model.
        """
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential(
            [
                model,
                GlobalAveragePooling2D(),
                Dense(512, activation='relu'),
                Dropout(dropout),
                Dense(self.num_classes, activation='sigmoid')
            ]
        )

        return model
    
    def _build_alexnet(
        self,
        input_shape: Tuple[int, int, int],
        dropout: float,
    ) -> Sequential:
        """
        Builds an AlexNet model with the given input shape, number of classes, dropout rate, and learning rate.

        Parameters:
            input_shape: The shape of the input data.
            num_classes: The number of classes for classification.
            dropout: The dropout rate.
            learning_rate: The learning rate for the optimizer.

        Returns:
            The compiled AlexNet model.
        """
        model = Sequential()
        model.add(
            Conv2D(
                96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation='relu',
                input_shape=input_shape
            )
        )
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Add more Conv2D and MaxPooling2D layers following the AlexNet architecture
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(dropout))
        return model

    def train_evaluate(
            self,
            learning_rate: float = 1e-4,
            dropout: float = 0.2,
            batch_size: int = 100,
            epochs: int = 10,
    ):
        """
        Trains and evaluates the model using cross-validation and the full dataset.

        Args:
            learning_rate (float): The learning rate for the optimizer (default: 1e-4).
            dropout (float): The dropout rate for regularization (default: 0.2).
            batch_size (int): The batch size for training (default: 100).
            epochs (int): The number of epochs for training (default: 10).

        Returns:
            float: The mean of the ks_stat_val metric calculated during cross-validation.
        """
        # Setting up TensorFlow and random seed for reproducibility
        self.setup_tensorflow()
        run = wandb.init(
            project=self.config.modelling.wandb_config.project_name,
            config={
                'learning_rate': learning_rate,
                'dropout': dropout,
                'batch_size': batch_size,
                'optimizer': str(self.optimizer),
                'epochs': epochs
            },
            save_code=True,
            tags=self.config.modelling.wandb_config.tags
        )

        # Cross Validation Process
        fold_number: int = 1
        fold_metrics_dict: Dict[str, List[float]] = {
            'roc_auc_train': [], 'roc_auc_val': [], 
            'ks_stat_train': [], 'ks_stat_val': [],
            'average_precision_train': [], 'average_precision_val': [],
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': []
        }
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.modelling.random_state)
        
        for train_split, val_split in stratified_kfold.split(self.X_train, self.y_train):
            logging.info("Runnning cross-validation: Fold", extra={"fold_number": fold_number})
            
            X_train_fold, y_train_fold = self.X_train[train_split], self.y_train[train_split]
            X_val_fold, y_val_fold = self.X_train[val_split], self.y_train[val_split]

            # Display the number of elements for each class before and after augmentation
            logging.info(f"Number of elements for class 0 before augmentation: {np.sum(y_train_fold == 0)}")
            logging.info(f"Number of elements for class 1 before augmentation: {np.sum(y_train_fold == 1)}")

            X_train_fold_augmented = self.augment_data(X_train_fold, num_augmentations=self.config.modelling.num_augmentations)
            y_train_fold_augmented = np.repeat(y_train_fold, self.config.modelling.num_augmentations + 1)
            logging.info("Augmentation data process finished", extra={"fold_number": fold_number})

            # Display the number of elements for each class after augmentation
            logging.info(f"Number of elements for class 0 after augmentation: {np.sum(y_train_fold_augmented == 0)}")
            logging.info(f"Number of elements for class 1 after augmentation: {np.sum(y_train_fold_augmented == 1)}")
            
            model = self.base_trainer(dropout, learning_rate)
            history = model.fit(
                X_train_fold_augmented,
                y_train_fold_augmented,
                steps_per_epoch=len(X_train_fold_augmented) // batch_size,
                validation_split=0.2,
                batch_size=int(batch_size), 
                epochs=int(epochs),
                callbacks=[wandb.keras.WandbCallback()]
            )

            # Calculate predictions from the model
            y_pred_proba_train_fold = model.predict(X_train_fold_augmented).flatten()
            y_pred_proba_val_fold = model.predict(X_val_fold).flatten()

            # Calculate metrics for the current fold
            fold_metrics_dict['roc_auc_train'].append(roc_auc_score(y_train_fold, model.predict(X_train_fold)))
            fold_metrics_dict['roc_auc_val'].append(roc_auc_score(y_val_fold, model.predict(X_val_fold)))
            fold_metrics_dict['ks_stat_train'].append(ks_2samp(y_pred_proba_train_fold[y_train_fold_augmented == 0], y_pred_proba_train_fold[y_train_fold_augmented == 1]).statistic)
            fold_metrics_dict['ks_stat_val'].append(ks_2samp(y_pred_proba_val_fold[y_val_fold == 0], y_pred_proba_val_fold[y_val_fold == 1]).statistic)
            fold_metrics_dict['average_precision_train'].append(skm.average_precision_score(y_train_fold_augmented, y_pred_proba_train_fold))
            fold_metrics_dict['average_precision_val'].append(skm.average_precision_score(y_val_fold, y_pred_proba_val_fold))
            fold_metrics_dict['train_loss'].append(history.history['loss'][-1])
            fold_metrics_dict['val_loss'].append(history.history['val_loss'][-1])
            fold_metrics_dict['train_accuracy'].append(history.history['accuracy'][-1])
            fold_metrics_dict['val_accuracy'].append(history.history['val_accuracy'][-1])
            
            fold_number += 1
        
        # Calculating and logging average CV metrics
        average_metrics = {f'average_{metric}': np.mean(values) for metric, values in fold_metrics_dict.items()}
        wandb.log({**average_metrics, **run.config})

        # Training on the full dataset
        logging.info("Training on the full dataset without cross-validation")
        model = self.base_trainer(dropout, learning_rate)
        history = model.fit(
            self.X_train, 
            self.y_train, 
            validation_split=0.2, 
            batch_size=batch_size, 
            epochs=epochs,
            callbacks=[wandb.keras.WandbCallback()]
        )

        # Calculate predictions from the model
        y_pred_proba_train = model.predict(self.X_train).flatten()
        y_pred_proba_test = model.predict(self.X_test).flatten()
         
        # Obtain the history losses (val/train) over the compilation
        training_metrics = {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
        
        # Calculate metrics.
        general_metrics = {
            'roc_auc_train': roc_auc_score(self.y_train, model.predict(self.X_train)),
            'roc_auc_test': roc_auc_score(self.y_test, model.predict(self.X_test)),
            'ks_test': ks_2samp(y_pred_proba_train[self.y_train == 0], y_pred_proba_train[self.y_train == 1]).statistic,
            'ks_train': ks_2samp(y_pred_proba_test[self.y_test == 0], y_pred_proba_test[self.y_test == 1]).statistic,
            'average_precision_train': skm.average_precision_score(self.y_train, y_pred_proba_train),
            'average_precision_test': skm.average_precision_score(self.y_test, y_pred_proba_test)
        }
        
        wandb.log({**training_metrics, **general_metrics})

        # Saving the model
        logging.info("Saving the model", extra={"model_name": run.name})
        model_path = os.path.join(self.config.modelling.models_dir, run.name)
        os.makedirs(model_path, exist_ok=True)
        model.save(os.path.join(model_path, 'model'))

        run.finish()

        return np.mean(average_metrics['ks_stat_val'])

    def initialize_train(self):
        """
        Initializes the training process by creating a BayesianOptimization object and maximizing it.

        :return: None
        """
        optimizer = BayesianOptimization(
            f=self.train_evaluate,
            pbounds=self.config.modelling.wandb_config.parameters,
            random_state=self.config.modelling.random_state,
            verbose=2
        )
        optimizer.maximize(
            init_points=self.config.modelling.bayesian_optimization.init_points,
            n_iter=self.config.modelling.bayesian_optimization.n_inter
            )
