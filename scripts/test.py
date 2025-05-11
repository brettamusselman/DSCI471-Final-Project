import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.cnn_model import CNN_Model
from data_loader import csv_image_label_generator
import pandas as pd

def load_dataset(csv_path, split="test", batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(csv_path, dataset=split),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    input_shape = (224, 224, 1)

    csv_path = "data/all_files_df.csv"
    df = pd.read_csv(csv_path)
    num_classes = df["label_two"].nunique()

    model = CNN_Model(input_shape=input_shape, num_classes=num_classes)
    model.model = tf.keras.models.load_model("checkpoints/final_model.h5")

    test_data = load_dataset(csv_path, split="test")
    model.evaluate_model(test_data)

if __name__ == "__main__":
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Testing will run on CPU.")
    main()
