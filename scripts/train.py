import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.cnn_model import CNN_Model
from data_loader import csv_image_label_generator

def load_dataset(csv_path, split="train", batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(csv_path, dataset=split),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    input_shape = (224, 224, 1)
    
    csv_path = "data/all_files_df.csv"

    # Dynamically get number of classes from label_two column
    import pandas as pd
    df = pd.read_csv(csv_path)
    num_classes = df["label_two"].nunique()

    train_data = load_dataset(csv_path, split="train")
    val_data = load_dataset(csv_path, split="val")

    model = CNN_Model(input_shape=input_shape, num_classes=num_classes)
    model.compile_model()

    log_dir = "logs"
    model.train_model(train_data, validation_data=val_data, epochs=20, log_dir=log_dir)

    model.model.save("checkpoints/final_model.h5")

if __name__ == "__main__":
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Training will run on CPU.")

    main()
