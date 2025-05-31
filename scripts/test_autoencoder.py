import tensorflow as tf
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.autoencoder import LungAutoencoderAnomaly
from data_loader import csv_image_label_generator

csv_path = "data/all_files_df.csv"
MODEL_PATH = "/mnt/c/Users/NesFa/repo/dsci471/logs/autoencoder_64/run_20250521-230946/checkpoints/best_autoencoder_val_loss_new.keras"
SAVE_HIST_PATH = "/mnt/c/Users/NesFa/repo/dsci471/results"
IMAGE_SHAPE = (224, 320, 1)
BATCH_SIZE = 32 

def load_test_data_with_labels(
    csv_path: str,
    image_shape: tuple[int, int, int],
    batch_size: int
):
    """
    Loads test dataset.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(
            csv_path=csv_path,
            dataset="test", 
            image_size=(458, 280),
            shuffle=False, 
            to_rgb=False,  
            binary_class=True,
            crop_image=True,
            equalize=True 
        ),
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_validation_normal_data(
    csv_path: str,
    image_shape: tuple[int, int, int],
    batch_size: int
):
    """
    Loads validation dataset.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(
            csv_path=csv_path,
            dataset="val",
            image_size=(458, 280),
            shuffle=False,
            to_rgb=False,
            binary_class=True, 
            with_class=0 # Filter for normal images only for calculating threshold
        ),
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return dataset.map(lambda img, lbl: img).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def plot_reconstruction_error_histogram(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str
):
    """
    Plots and saves a histogram of reconstruction errors
    """
    plt.figure(figsize=(12, 7))
    
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]
    
    if len(normal_errors) > 0:
        sns.histplot(normal_errors, color="skyblue", label="Normal (True Label 0) Errors", kde=True, stat="density", common_norm=False)
    if len(anomaly_errors) > 0:
        sns.histplot(anomaly_errors, color="salmon", label="Anomaly (True Label 1) Errors", kde=True, stat="density", common_norm=False)
    
    plt.axvline(threshold, color="green", linestyle="--", linewidth=2, label=f"Anomaly Threshold ({threshold:.4f})")
    
    plt.title("Distribution of Reconstruction Errors on Test Set")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved reconstruction error histogram to {save_path}")
    plt.close()


def main():
    print(f"Loading autoencoder model from: {MODEL_PATH}")
    autoencoder_model = LungAutoencoderAnomaly(
        input_shape=IMAGE_SHAPE, 
        model_path=MODEL_PATH,
        latent_dim=16
    )

    val_normal_images_ds = load_validation_normal_data(csv_path, IMAGE_SHAPE, BATCH_SIZE)
    val_normal_errors = autoencoder_model.get_reconstruction_error(val_normal_images_ds)

    threshold = np.percentile(val_normal_errors, 95) # getting threshold from validation data
    test_ds_with_labels = load_test_data_with_labels(csv_path, IMAGE_SHAPE, BATCH_SIZE)

    # Collect all test images and true labels
    all_test_images = []
    all_test_labels = []
    for images_batch, labels_batch in test_ds_with_labels:
        all_test_images.append(images_batch.numpy())
        all_test_labels.append(labels_batch.numpy())

    all_test_images_np = np.concatenate(all_test_images, axis=0)
    all_test_labels_np = np.concatenate(all_test_labels, axis=0)

    test_reconstruction_errors = autoencoder_model.get_reconstruction_error(all_test_images_np)

    predicted_anomalies = (test_reconstruction_errors > threshold).astype(int)

    accuracy = accuracy_score(all_test_labels_np, predicted_anomalies)
    print(f"\n--- Anomaly Detection Performance on Test Set ---")
    print(f"Threshold: {threshold:.6f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")

    target_names = ['Normal (Class 0)', 'Anomaly (Class 1)']
    print(classification_report(all_test_labels_np, predicted_anomalies, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_test_labels_np, predicted_anomalies)
    print(cm)

    histogram_save_path = os.path.join(SAVE_HIST_PATH, "test_reconstruction_errors_histogram.png")
    plot_reconstruction_error_histogram(
        test_reconstruction_errors,
        all_test_labels_np,
        threshold,
        histogram_save_path
    )


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected and memory growth set: {gpus[0].name}")
        except RuntimeError as e:
            print(f"Error during GPU setup: {e}")
    else:
        print("No GPU detected. Evaluation will run on CPU.")
    main()