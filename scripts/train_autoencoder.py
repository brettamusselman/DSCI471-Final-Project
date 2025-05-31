import tensorflow as tf
import sys
import os
import pandas as pd
from tensorflow.keras import layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.autoencoder import LungAutoencoderAnomaly
from data_loader import csv_image_label_generator

csv_path = "data/all_files_df.csv"

data_augmentation_pipeline = tf.keras.Sequential([
    layers.RandomTranslation(0.05, 0.05, name="rand_translate"),
    layers.RandomZoom(0.05, name="rand_zoom"),
    layers.RandomRotation(0.05, fill_mode="reflect", name="rand_rot"),
    
    layers.GaussianNoise(0.03, name="gaussian_noise"),
], name="data_augmentation_pipeline")


def augment_data(image1):
    """Applies data augmentation to an image"""
    image = data_augmentation_pipeline(image1, training=True)
    return image, image

def autoencoder_image_generator(csv_path: str, dataset: str = "train",
                                shuffle: bool = True, to_rgb: bool = False,
                                image_size: tuple[int, int] = (384, 246), with_class = 0
                               ):

    image_label_gen = csv_image_label_generator(
        csv_path=csv_path,
        dataset=dataset,
        shuffle=shuffle,
        to_rgb=to_rgb,
        image_size=image_size,
        with_class=with_class,
        crop_image=True,
        equalize=True
    )

    for image, _ in image_label_gen:
        yield image, image

def load_dataset(csv_path, split="train", batch_size=32, binary_class=False, shuffle=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: autoencoder_image_generator(csv_path, dataset=split, image_size=(458, 280), with_class=0, to_rgb=False),
        output_signature=(
            tf.TensorSpec(shape=(224, 320, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(224, 320, 1), dtype=tf.float32)
        )
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    input_shape = (224, 320, 1)

    train_data = load_dataset(csv_path, split="train")
    val_data = load_dataset(csv_path, split="val")

    model = LungAutoencoderAnomaly(
        input_shape=input_shape,
        latent_dim=16
    )

    model.compile_model(
        optimizer_name='adam', 
        learning_rate=0.001,
        loss=tf.keras.losses.MeanSquaredError()
    )

    model.train_model(
        train_data_healthy=train_data,
        validation_data_healthy=val_data,
        epochs=100,
        log_dir="logs/autoencoder_64",
    )

    model.save_model("checkpoints/final_model_autoencoder .h5")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Training will run on CPU.")

    main()
