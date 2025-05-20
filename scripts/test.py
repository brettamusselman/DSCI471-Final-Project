import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transfer_resnet_model import TransferResNet50V2
from data_loader import csv_image_label_generator
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_dataset(csv_path, split="test", batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_image_label_generator(csv_path, dataset=split, to_rgb=True, image_size=(1348, 987), shuffle=False, binary_class=False),
        output_signature=(
            tf.TensorSpec(shape=(987, 1348, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """
    Saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    plt.savefig(save_path)

def main():
    csv_path = "data/all_files_df.csv"
    df = pd.read_csv(csv_path)

    loaded_resnet = TransferResNet50V2(model_path="./logs/resnet50v2_transfer_pneumonia/run_20250518-083717/checkpoints/best_transfer_model_val_loss.keras")
    loaded_resnet.model.summary()

    test_data = load_dataset(csv_path, split="test")
    y_true = [label.numpy() for _, label in test_data.unbatch()]
    y_true = np.array(y_true).argmax(axis=-1) # For multiclass, remove for binary

    predicted = loaded_resnet.predict(image_data=test_data)
    predicted_classes = predicted.argmax(axis=-1) # For multiclass
    # predicted_classes = (predicted[:, 0] >= 0.5).astype(int) #For binary

    class_names = ["Normal", "Bacterial", "Viral"] # For multiclass
    # class_names = ["Normal", "Pneumonia"] # For binary

    plot_confusion_matrix(y_true, predicted_classes, class_names, save_path="resnet50v2_transfer_pneumonia_20250518-083717_multiclass_best.png")

if __name__ == "__main__":
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Testing will run on CPU.")
    main()
