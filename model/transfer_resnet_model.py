import tensorflow as tf
from tensorflow.keras import layers, Input, Model, regularizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import logging
from typing import Tuple, List, Union, Optional, Dict
import os
import datetime

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TransferResNet50V2:
    def __init__(self, input_shape: Tuple[int, int, int] = None, num_classes: int = None, model_path: str = None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.base_model: Optional[Model] = None
            self.model: Optional[Model] = None
            self.metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None
            self._build_model()

    def _build_model(self):
        """
        Builds the transfer learning model.
        Loads ResNet50V2 pre-trained on ImageNet, freezes its layers,
        and adds a custom classification head.
        """
        try:
            self.base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape,
                pooling=None
            )

            self.base_model.trainable = False

            inputs = Input(shape=self.input_shape, name="input_layer")
            x = self.base_model(inputs, training=False)
            
            # Custom head
            x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
            x = layers.Dense(256, activation='relu', name="custom_dense_1", kernel_initializer='he_normal')(x)
            # x = layers.BatchNormalization(name="custom_bn_1")(x)
            # x = layers.Activation('relu', name="custom_relu_1")(x)
            x = layers.Dropout(0.2, name="custom_dropout")(x) # Dropout for regularization
            outputs = layers.Dense(self.num_classes, activation='softmax', name="custom_predictions", kernel_initializer='glorot_uniform')(x)
            # outputs = layers.Dense(1, activation='sigmoid', name="custom_predictions", kernel_initializer='glorot_uniform')(x) # for binary classification

            self.model = Model(inputs=inputs, outputs=outputs, name="TransferResNet50V2")
            logger.info("Transfer learning model with ResNet50V2 base built successfully.")
            self.model.summary(print_fn=logger.info)

        except Exception as e:
            logger.error(f"Error building transfer learning model: {e}", exc_info=True)
            raise

    def compile_model(self, 
                      optimizer_name: str = 'adam', 
                      learning_rate: float = 0.001, 
                      loss = tf.keras.losses.CategoricalCrossentropy(), 
                      metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None):
        if metrics is None:
            metrics = ['accuracy']
        self.metrics = metrics
        
        try:
            if optimizer_name.lower() == 'adam':
                optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name.lower() == 'adamw':
                optimizer_instance = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
            elif optimizer_name.lower() == 'sgd':
                optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

            self.model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)
            logger.info(f"Model compiled with optimizer: {optimizer_name}, LR: {learning_rate}, loss: {loss}.")
        except Exception as e:
            logger.error(f"Error compiling model: {e}", exc_info=True)
            raise

    def train_model(self, 
                    train_data: tf.data.Dataset, 
                    validation_data: tf.data.Dataset, 
                    epochs_feature_extraction: int = 10,
                    epochs_fine_tuning: int = 10,
                    fine_tune_layers: int = 20, 
                    fine_tune_learning_rate: float = 1e-5,
                    log_dir: str = "logs_transfer", 
                    class_weights: Optional[Dict[int, float]] = None,
                    ):
        """
        Trains the model in two stages:
        1. Trains only the custom head with the base model frozen.
        2. Unfreezes some layers of the base model.
        """

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(current_run_log_dir, exist_ok=True)

        tensorboard_callback = TensorBoard(
            log_dir=current_run_log_dir, 
            histogram_freq=1, 
            profile_batch=0 
        )
        
        checkpoint_dir = os.path.join(current_run_log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(checkpoint_dir, "best_transfer_model_val_loss.keras")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=False, verbose=1
        )

        reduceLr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=1,
                mode='min',
                min_delta=0.001,
                cooldown=0,
                min_lr=1e-5,
            )

        callbacks_list = [tensorboard_callback, model_checkpoint_callback, reduceLr]

        # Training head
        if epochs_feature_extraction > 0:
            self.base_model.trainable = False 

            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs_feature_extraction,
                callbacks=callbacks_list,
                class_weight=class_weights
            )

        # Training last few layers in base model
        if epochs_fine_tuning > 0 and fine_tune_layers > 0:            
            self.base_model.trainable = True

            num_base_layers = len(self.base_model.layers)
            freeze_until_layer = num_base_layers - fine_tune_layers

            for i, layer in enumerate(self.base_model.layers):
                if i < freeze_until_layer:
                    layer.trainable = False
                else:
                    layer.trainable = True
            
            logger.info(f"Fine-tuning: {len(self.base_model.trainable_weights)} trainable weights in base model.")

            current_loss = self.model.loss
            current_metrics = self.metrics
            tune_optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate)

            self.model.compile(optimizer=tune_optimizer, loss=current_loss, metrics=current_metrics) # recompile with lower LR

            reduceLr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=1,
                mode='min',
                min_delta=0.001,
                cooldown=0,
                min_lr=1e-7,
            )

            callbacks_list = [tensorboard_callback, model_checkpoint_callback, reduceLr]

            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs_feature_extraction + epochs_fine_tuning,
                initial_epoch=epochs_feature_extraction,
                callbacks=callbacks_list,
                class_weight=class_weights
            )
        
        logger.info(f"Training completed. TensorBoard logs in: {current_run_log_dir}")


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

