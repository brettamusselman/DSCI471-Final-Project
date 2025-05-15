import tensorflow as tf
from tensorflow.keras import layers, Input 
from tensorflow.keras.models import Model
import logging
from typing import Tuple, List, Union, Optional, Dict
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ResNet_Model():
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the ResNet model.

        Parameters:
        - input_shape: Shape of the input images (height, width, channels).
        - num_classes: Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _basic_residual_block(self, 
                                 x_input: tf.Tensor, 
                                 filters: int, 
                                 kernel_size: Tuple[int, int] = (3, 3), 
                                 strides: Tuple[int, int] = (1, 1),
                                 block_name: Optional[str] = None) -> tf.Tensor:
        """
        Creates a ResNet basic residual blockwith pre-activation.

        Parameters:
        - x_input: Input tensor.
        - filters: Number of filters for the convolutional layers in the block.
        - kernel_size: Kernel size for the convolutional layers.
        - strides: Strides for the first convolutional layer in the block and for the shortcut if projection is needed.
        - block_name: Optional string for naming layers within this block.

        Returns:
        - Output tensor of the residual block.
        """
        if block_name is None:
            block_name = f"b_res_block_{tf.keras.backend.get_uid('b_res_block')}"

        preact = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=f'{block_name}_preact_bn')(x_input)
        preact = layers.Activation('relu', name=f'{block_name}_preact_relu')(preact)

        if strides != (1, 1) or x_input.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same',
                                     kernel_initializer='he_normal', 
                                     name=f'{block_name}_shortcut_conv')(preact)
        else:
            shortcut = x_input # Identity shortcut

        y = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                          kernel_initializer='he_normal', 
                          name=f'{block_name}_1_conv')(preact)
        y = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=f'{block_name}_1_bn')(y)
        y = layers.Activation('relu', name=f'{block_name}_1_relu')(y)

        y = layers.Conv2D(filters, kernel_size, strides=(1,1), padding='same',
                          kernel_initializer='he_normal', 
                          name=f'{block_name}_2_conv')(y)
        
        output = layers.add([shortcut, y], name=f'{block_name}_add')
        return output

    def _stack_basic(self, x: tf.Tensor, filters: int, num_blocks: int, 
                        stage_name: str, strides_first_block: Tuple[int, int] = (2,2)):
        """Helper function to stack ResNetV2 basic residual blocks."""
        x = self._basic_residual_block(x, filters, strides=strides_first_block,
                                          block_name=f'{stage_name}_block1')
        for i in range(2, num_blocks + 1):
            x = self._basic_residual_block(x, filters, strides=(1,1),
                                              block_name=f'{stage_name}_block{i}')
        return x

    def _build_model(self) -> Model:
        """
        Build the ResNet model architecture.
        Number of blocks: [3, 4, 4, 3] 
        Filters per stage: [64, 128, 256, 512].

        Returns:
        - model: Keras Model instance.
        """
        try:
            inputs = Input(shape=self.input_shape)

            x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', 
                              kernel_initializer='he_normal', name='conv1_conv')(inputs)
            x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)
            x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1_pool')(x)

            x = self._stack_basic(x, filters=64, num_blocks=3, stage_name='conv2', strides_first_block=(1,1))

            x = self._stack_basic(x, filters=128, num_blocks=4, stage_name='conv3', strides_first_block=(2,2))

            x = self._stack_basic(x, filters=256, num_blocks=4, stage_name='conv4', strides_first_block=(2,2))
            
            x = self._stack_basic(x, filters=512, num_blocks=3, stage_name='conv5', strides_first_block=(2,2))

            # Post-activation before Global Average Pooling
            x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='post_bn')(x)
            x = layers.Activation('relu', name='post_relu')(x)
            
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            outputs = layers.Dense(self.num_classes, activation='softmax', 
                                   kernel_initializer='glorot_uniform', name='predictions')(x)

            model = Model(inputs=inputs, outputs=outputs, name="resnet_model")
            logger.info("ResNet model built successfully.")
            return model
        except Exception as e:
            logger.error(f"Error building ResNet model: {e}", exc_info=True)
            raise

    def compile_model(self, 
                      optimizer_name: str = 'adam', 
                      learning_rate: float = 0.001, 
                      loss: str = 'sparse_categorical_crossentropy', 
                      metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None):
        """
        Compile the ResNet model with a specified optimizer and initial learning rate.

        Parameters:
        - optimizer_name: Name of the optimizer to use.
        - learning_rate: Initial learning rate for the optimizer.
        - loss: Loss function to use for training.
        - metrics: List of metrics to evaluate during training.
        """
        if metrics is None:
            metrics = ['accuracy']
        
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

    def train_model(self, 
                    train_data: tf.data.Dataset, 
                    validation_data: tf.data.Dataset, 
                    epochs: int = 20, 
                    log_dir: str = "logs",
                    class_weights: Optional[Dict[int, float]] = None
                    ):
        """
        Train the ResNet model

        Parameters:
        - train_data: Training data (tf.data.Dataset).
        - validation_data: Validation data (tf.data.Dataset).
        - epochs: Number of epochs to train for.
        - log_dir: Directory to save TensorBoard logs.
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            current_log_dir = os.path.join(log_dir, timestamp)
            os.makedirs(current_log_dir, exist_ok=True)
            tensorboard_callback = TensorBoard(log_dir=current_log_dir, histogram_freq=1, profile_batch=0)

            current_checkpoint_dir = os.path.join(current_log_dir, "checkpoint")
            os.makedirs(current_checkpoint_dir, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                filepath=current_checkpoint_dir,
                monitor='val_loss',       
                mode='min',            
                save_best_only=True,    
                save_weights_only=False,  
                verbose=1        
            )
            
            callbacks_list = [tensorboard_callback, checkpoint_callback]

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
            logger.error(f"Error training model: {e}", exc_info=True) # Added exc_info for more detailed error logging
            raise

    def save_model(self, model_path: str):
        """
        Save the ResNet model to a specified path.
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            logger.info(f"Model saved at {model_path}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def evaluate_model(self, test_data: tf.data.Dataset, steps: int = None):
        """
        Evaluate the ResNet model.
        """
        try:
            logger.info("Evaluating model...")
            evaluation = self.model.evaluate(test_data, steps=steps, verbose=1)
            if self.model.metrics_names:
                 for metric_name, value in zip(self.model.metrics_names, tf.nest.flatten(evaluation)): 
                    logger.info(f"Evaluation - {metric_name}: {value:.4f}")
            else:
                logger.info(f"Model evaluation results (raw): {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def predict(self, image_data: Union[tf.Tensor, tf.data.Dataset], batch_size: int = 32):
        """
        Make a prediction using the ResNet model.
        """
        try:
            if isinstance(image_data, tf.Tensor) and len(image_data.shape) == 3:
                image_data = tf.expand_dims(image_data, axis=0)
            
            logger.info("Making predictions...")
            predictions = self.model.predict(image_data, batch_size=batch_size, verbose=0)
            logger.info(f"Model predictions generated with shape: {predictions.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error making prediction with model: {e}")
            raise
