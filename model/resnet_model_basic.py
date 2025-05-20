import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
import logging
from typing import Tuple, Optional, Dict, Union
import os
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import datetime

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ResNet_Model():
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the ResNet  model.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: Tuple[int, int] = (3, 3), strides: Tuple[int, int] = (1, 1)) -> tf.Tensor:
        """
        Creates a residual block.
        """
        y = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        y = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(y) # Second conv
        y = layers.BatchNormalization()(y)

        # Shortcut connection
        shortcut = x
        if strides != (1, 1) or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(x)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.Activation('relu')(y) # Final activation after addition
        return y

    def _build_model(self) -> Model:
        """
        Build the ResNet-like model architecture.
        """
        try:
            inputs = Input(shape=self.input_shape)


            x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # Further downsampling

            # Residual Blocks - Stage 1 (64 filters)
            x = self._residual_block(x, filters=64)
            x = self._residual_block(x, filters=64)

            # Residual Blocks - Stage 2 (128 filters)
            x = self._residual_block(x, filters=128, strides=(2, 2)) 
            x = self._residual_block(x, filters=128)

            # Residual Blocks - Stage 3 (256 filters)
            x = self._residual_block(x, filters=256, strides=(2, 2))
            x = self._residual_block(x, filters=256)
            
            # can be added for a deeper model
            # x = self._residual_block(x, filters=512, strides=(2, 2))
            # x = self._residual_block(x, filters=512)

            x = layers.GlobalAveragePooling2D()(x)
            
            x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
            x = layers.Dropout(0.5)(x)
            
            # Output
            outputs = layers.Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
            # outputs = layers.Dense(1, activation='sigmoid', name="custom_predictions", kernel_initializer='glorot_uniform')(x) # For binary classification

            model = Model(inputs=inputs, outputs=outputs, name="resnet_model")
            logger.info("ResNet model built successfully.")
            return model
        except Exception as e:
            logger.error(f"Error building ResNet-like model: {e}")
            raise

    def compile_model(self, optimizer_name: str = 'adam', loss: str = 'categorical_crossentropy', metrics: list = ['accuracy'], learning_rate: float = 0.001):
        """
        Compile the ResNet-like model.
        """
        try:
            if optimizer_name.lower() == 'adam':
                optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name.lower() == 'sgd':
                optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)

            self.model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)
            logger.info(f"Model compiled successfully with optimizer: {optimizer_name}, initial LR: {learning_rate}, loss: {loss}.")
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise

    def train_model(self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset, 
                    epochs: int = 20, log_dir: str = "logs", class_weights: Optional[Dict[int, float]] = None):
        """
        Train the ResNet model with TensorBoard callbacks.
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            current_log_dir = os.path.join(log_dir, timestamp)
            os.makedirs(current_log_dir, exist_ok=True)
            tensorboard_callback = TensorBoard(log_dir=current_log_dir, histogram_freq=1, profile_batch=0)

            checkpoint_dir = os.path.join(current_log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_filepath = os.path.join(checkpoint_dir, "best_resnet_model_val_loss.keras")

            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',       
                mode='min',            
                save_best_only=True,    
                save_weights_only=False,  
                verbose=1        
            )

            reduceLr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=1,
                mode='min',
                min_delta=0.001,
                cooldown=0,
                min_lr=1e-6,
            )
            
            callbacks_list = [tensorboard_callback, checkpoint_callback, reduceLr]

            logger.info(f"Starting model training for {epochs} epochs. Logs will be saved to {current_log_dir}")
            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks_list,
                class_weight=class_weights
            )
            logger.info("Model training completed.")
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True) 
            raise

    def save_model(self, model_path: str):
        """Saves the Keras model."""
        try:
            save_dir = os.path.dirname(model_path)
            if save_dir and not os.path.exists(save_dir): 
                os.makedirs(save_dir, exist_ok=True)
            self.model.save(model_path)
            logger.info(f"Model saved at {model_path}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise

    def evaluate_model(self, test_data: tf.data.Dataset):
        """Evaluates the Keras model."""
        try:
            evaluation = self.model.evaluate(test_data, verbose=1)
            logger.info(f"CNN model evaluation results: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            raise

    def predict(self, image_data: Union[tf.Tensor, tf.data.Dataset], batch_size: int = 32):
        """Makes predictions using the Keras model."""
        try:
            predictions = self.model.predict(image_data, batch_size=batch_size, verbose=1)
            return predictions
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            raise

