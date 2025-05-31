import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import recall_score, precision_score
import logging
from typing import Tuple, List, Union, Optional, Dict, Callable, Iterator
import os
import datetime
import numpy as np

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LungAutoencoderAnomaly:
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1), latent_dim: int = 32, model_path: str = None):
        if model_path:
            try:
                self.autoencoder = tf.keras.models.load_model(model_path)
                self.input_shape = self.autoencoder.input_shape[1:]
                self.latent_dim = latent_dim
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
                raise
        else:
            self.input_shape = input_shape
            self.latent_dim = latent_dim
            self.encoder: Optional[Model] = None
            self.decoder: Optional[Model] = None
            self.autoencoder: Optional[Model] = None
            self.metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None
            self._build_model()


    def _build_model(self):
        inp = tf.keras.Input(self.input_shape, name="ae_input")        # (224,320,1)
        x_color = tf.keras.layers.Conv2D(3, 1, padding="same",
                                        name="color_adapter")(inp)        # (224,320,3)

        vgg_base = tf.keras.applications.VGG16(include_top=False,
                                                weights="imagenet",
                                                input_shape=(224, 320, 3)) 
        vgg_base.trainable = False

        x = vgg_base(x_color)                                        # (7,10,512)

        # bottleneck
        shape_before_flat = tf.keras.backend.int_shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)
        z = tf.keras.layers.Dense(self.latent_dim, activation="relu",
                                name="latent")(x)
        self.encoder = tf.keras.Model(inp, z, name="encoder")

        # decoder 
        dec_in = tf.keras.Input((self.latent_dim,), name="decoder_input")
        x = tf.keras.layers.Dense(np.prod(shape_before_flat), activation="relu")(dec_in)
        x = tf.keras.layers.Reshape(shape_before_flat)(x)

        def deconv(x, f):
            x = tf.keras.layers.Conv2DTranspose(f, 3, padding="same",
                                                kernel_initializer="he_normal")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return tf.keras.layers.ReLU()(x)

        # Upsample
        x = tf.keras.layers.UpSampling2D()(x)              
        x = deconv(x, 512)
        x = deconv(x, 512)
        x = tf.keras.layers.UpSampling2D()(x)               
        x = deconv(x, 512)
        x = deconv(x, 512)
        x = deconv(x, 512)
        x = tf.keras.layers.UpSampling2D()(x)               
        x = deconv(x, 256)
        x = deconv(x, 256)
        x = deconv(x, 256)
        x = tf.keras.layers.UpSampling2D()(x)               
        x = deconv(x, 128) 
        x = deconv(x, 128)
        x = tf.keras.layers.UpSampling2D()(x)         
        x = deconv(x,  64)
        x = deconv(x,  64)

        out = tf.keras.layers.Conv2D(1, 3, padding="same",
                                    activation="sigmoid",
                                    name="decoder_output")(x)

        self.decoder = tf.keras.Model(dec_in, out, name="decoder")
        self.autoencoder = tf.keras.Model(inp, self.decoder(self.encoder(inp)),
                                        name="autoencoder")

    def compile_model(self,
                      optimizer_name: str = 'adam',
                      learning_rate: float = 0.001,
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None):
        if metrics is None:
            metrics = ['mse']
        self.metrics = metrics

        try:
            if optimizer_name.lower() == 'adam':
                optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name.lower() == 'adamw':
                optimizer_instance = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
            elif optimizer_name.lower() == 'sgd':
                optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

            self.autoencoder.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)
            logger.info(f"Autoencoder compiled with optimizer: {optimizer_name}, LR: {learning_rate}, loss: {loss}.")
        except Exception as e:
            logger.error(f"Error compiling autoencoder: {e}", exc_info=True)
            raise

    def train_model(self,
                    train_data_healthy: tf.data.Dataset,
                    validation_data_healthy: Optional[tf.data.Dataset] = None, 
                    epochs: int = 50,
                    batch_size: Optional[int] = 32, 
                    log_dir: str = "logs_autoencoder",
                    callbacks_list: Optional[List[tf.keras.callbacks.Callback]] = []):
        """
        Trains the autoencoder model.
        """
        if self.autoencoder is None:
            logger.error("Model has not been built or loaded.")
            return

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
        checkpoint_filepath = os.path.join(checkpoint_dir, "best_autoencoder_val_loss.keras")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_pneumonia_recall',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=30,
            verbose=1,
            mode='min',
            min_delta=0.0001,
            min_lr=1e-7
        )

        callbacks_list += [tensorboard_callback, model_checkpoint_callback, reduce_lr_callback]

        logger.info(f"Starting autoencoder training for {epochs} epochs.")

        history = self.autoencoder.fit(
            train_data_healthy,
            validation_data=validation_data_healthy,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        logger.info(f"Autoencoder training completed. TensorBoard logs in: {current_run_log_dir}")
        return history

    def get_reconstruction_error(self, data_x: Union[tf.Tensor, np.ndarray, tf.data.Dataset]):
        """
        Calculates the mean squared reconstruction error for each input sample.
        """
        if isinstance(data_x, tf.data.Dataset):
            reconstructed_x = self.autoencoder.predict(data_x)
            errors = []
            original_images = []
            for batch in data_x:
                img_batch = batch[0] if isinstance(batch, tuple) else batch
                original_images.append(img_batch.numpy())
            original_images_np = np.concatenate(original_images, axis=0)
            if reconstructed_x.shape[0] != original_images_np.shape[0]:
                 reconstructed_x = self.autoencoder.predict(original_images_np)
        else:
            original_images_np = data_x.numpy() if isinstance(data_x, tf.Tensor) else data_x
            reconstructed_x = self.autoencoder.predict(original_images_np)

        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        errors = loss_fn(tf.reshape(original_images_np, (len(original_images_np), -1)),
                                                    tf.reshape(reconstructed_x, (len(reconstructed_x), -1)))
        return errors.numpy()


    def save_model(self, model_path: str):
        """Saves the autoencoder model."""
        try:
            save_dir = os.path.dirname(model_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            self.autoencoder.save(model_path)
            logger.info(f"Autoencoder model saved at {model_path}.")
        except Exception as e:
            logger.error(f"Error saving autoencoder model: {e}", exc_info=True)
            raise