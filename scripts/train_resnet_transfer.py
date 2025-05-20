import tensorflow as tf
import sys
import os
import pandas as pd
from tensorflow.keras import layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transfer_resnet_model import TransferResNet50V2
from data_loader import csv_image_label_generator

data_augmentation_pipeline = tf.keras.Sequential([
    layers.RandomRotation(0.1, name="random_rotation_map"),
    layers.RandomZoom(0.1, name="random_zoom_map"),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1, name="random_translation_map"),
    layers.RandomContrast(0.1, name="random_contrast_map")
], name="data_augmentation_pipeline")


def augment_data(image, label):
    """Applies data augmentation to an image"""
    image = data_augmentation_pipeline(image, training=True)
    return image, label

def load_dataset(csv_path, split="train", batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(csv_path, dataset=split, image_size=(1348//2, 987//2), to_rgb=True),
        output_signature=(
            tf.TensorSpec(shape=(987//2, 1348//2, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32)
        )
    )

    if split == "train":
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        print(f"Data augmentation applied to '{split}' dataset.")

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    input_shape = (987//2, 1348//2, 3)
    
    csv_path = "data/all_files_df.csv"

    df = pd.read_csv(csv_path)
    num_classes = df["label_two"].nunique()

    train_data = load_dataset(csv_path, split="train")
    val_data = load_dataset(csv_path, split="val")

    model = TransferResNet50V2(input_shape=input_shape, num_classes=num_classes)

    EPOCHS_FEATURE_EXTRACTION = 50
    EPOCHS_FINE_TUNING = 50   
    FINE_TUNE_LAYERS = 10      
    INITIAL_LR_HEAD = 1e-3
    FINE_TUNE_LR = 1e-5

    model.compile_model(
        optimizer_name='sgd', 
        learning_rate=INITIAL_LR_HEAD,
        metrics=[tf.keras.metrics.categorical_accuracy, 
                 tf.keras.metrics.Precision(class_id=0, name="normal_precision"),
                 tf.keras.metrics.Recall(class_id=0, name="normal_recall"),
                 tf.keras.metrics.Precision(class_id=1, name="bacterial_precision"),
                 tf.keras.metrics.Recall(class_id=1, name="bacterial_recall"),
                 tf.keras.metrics.Precision(class_id=2, name="viral_precision"),
                 tf.keras.metrics.Recall(class_id=2, name="viral_recall")]    
    )

    total_samples = 1266 + 2224 + 1194

    class_weights_dict = {
        0: total_samples / (num_classes * 1266),
        1: total_samples / (num_classes * 2224),
        2: total_samples / (num_classes * 1194)
    }

    model.train_model(
        train_data,
        val_data,
        epochs_feature_extraction=EPOCHS_FEATURE_EXTRACTION,
        epochs_fine_tuning=EPOCHS_FINE_TUNING,
        fine_tune_layers=FINE_TUNE_LAYERS,
        fine_tune_learning_rate=FINE_TUNE_LR,
        log_dir="logs/resnet50v2_transfer_pneumonia",
        class_weights=class_weights_dict

    )

    model.model.save("checkpoints/final_model_tl.h5")

if __name__ == "__main__":
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Training will run on CPU.")

    main()
