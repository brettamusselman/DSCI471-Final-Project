import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Define the CNN model
class CNN_Model():
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the CNN model.

        Parameters:
        - input_shape: Shape of the input images (height, width, channels).
        - num_classes: Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Build the CNN model architecture.

        Returns:
        - model: Keras Model instance.
        """
        try:
            model = models.Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
            logger.info("CNN model built successfully.")
            return model
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise

    def compile_model(self, optimizer: str = 'adam', loss: str = 'sparse_categorical_crossentropy', metrics: list = ['accuracy']):
        """
        Compile the CNN model.

        Parameters:
        - optimizer: Optimizer to use for training.
        - loss: Loss function to use for training.
        - metrics: List of metrics to evaluate during training.
        """
        try:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            logger.info("CNN model compiled successfully.")
        except Exception as e:
            logger.error(f"Error compiling CNN model: {e}")
            raise

    def train_model(self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset, epochs: int = 10, log_dir: str = "logs"):
        """
        Train the CNN model with early stopping and TensorBoard callbacks.

        Parameters:
        - train_data: Training data.
        - validation_data: Validation data.
        - epochs: Number of epochs to train for.
        - log_dir: Directory to save TensorBoard logs.
        """
        try:
            # Create a timestamped subdirectory for logs
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=os.path.join(log_dir, timestamp), histogram_freq=1)

            # Define early stopping
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=[early_stopping_callback, tensorboard_callback]
            )
            logger.info("CNN model trained successfully with callbacks.")
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            raise

    def save_model(self, model_path: str):
        """
        Save the CNN model to a specified path.

        Parameters:
        - model_path: Path to save the model.
        """
        try:
            self.model.save(model_path)
            logger.info(f"CNN model saved at {model_path}.")
        except Exception as e:
            logger.error(f"Error saving CNN model: {e}")
            raise

    def evaluate_model(self, test_data: tf.data.Dataset):
        """
        Evaluate the CNN model.

        Parameters:
        - test_data: Test data.
        """
        try:
            evaluation = self.model.evaluate(test_data)
            logger.info(f"CNN model evaluation results: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating CNN model: {e}")
            raise

    def predict(self, image: tf.Tensor):
        """
        Make a prediction using the CNN model.

        Parameters:
        - image: Input image tensor.
        """
        try:
            prediction = self.model.predict(tf.expand_dims(image, axis=0))
            logger.info(f"CNN model prediction: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction with CNN model: {e}")
            raise