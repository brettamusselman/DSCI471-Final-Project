import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import io
import logging
import os
import pandas as pd
import cv2

logger = logging.getLogger(__name__)

BASE_IMAGE_FOLDER = "/mnt/c/Users/NesFa/Desktop/trainingFiles"

def histogram_equalization_cv(image_np):
    """
    Applies histogram equalization using OpenCV.
    """
    
    equalized_image_np = cv2.equalizeHist(image_np)
    
    return equalized_image_np


def load_image_tensor_from_url(url: str, image_size: tuple[int, int] = (1348, 987), equalize: bool = False) -> tf.Tensor:
    """
    Load an image from a URL and convert it to a TensorFlow tensor.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("L")
        image = image.resize(image_size)
        image_np = np.array(image)

        if equalize:
            image_np = histogram_equalization_cv(image_np)
        return tf.convert_to_tensor(image_np, dtype=tf.uint8)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image from {url}: {e}")
        raise
    except (IOError, Image.UnidentifiedImageError) as e:
        logging.error(f"Error processing image from {url}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading image from {url}: {e}")
        raise

def symmetric_crop(image: tf.Tensor,
                   pct_left_right: float = 0.15,
                   pct_top_bottom: float = 0.10) -> tf.Tensor:

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    crop_lr  = tf.cast(tf.round(tf.cast(w, tf.float32) * pct_left_right), tf.int32)
    crop_tb  = tf.cast(tf.round(tf.cast(h, tf.float32) * pct_top_bottom), tf.int32)

    return tf.image.crop_to_bounding_box(
        image,
        offset_height=crop_tb,
        offset_width=crop_lr,
        target_height=h - 2 * crop_tb,
        target_width=w - 2 * crop_lr,
    )

def csv_image_label_generator(csv_path: str, dataset: str = "train", binary_class: bool = False, shuffle: bool = True, to_rgb: bool = False, image_size: tuple[int, int] = (1348, 987), with_class: int = None, crop_image: bool = False, equalize: bool = False) -> tf.data.Dataset:
    """
    Generator yielding (image_tensor, label) pairs from a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset]

    if with_class is not None:
        df = df[df["label_two"] == with_class]

    num_classes = df["label_two"].nunique()
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    for _, row in df.iterrows():
        try:
            image = load_image_tensor_from_url(row["public_url"], image_size=image_size, equalize=equalize)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, -1)

            if (crop_image):
                image = symmetric_crop(image)

            label = int(row["label_two"])

            if binary_class:
                if label != 0:
                    label = 1
            else:
                label = tf.one_hot(label, depth=num_classes, dtype=tf.int32)

            if to_rgb:
                image = tf.image.grayscale_to_rgb(image)

            yield image, label
        except Exception as e:
            logger.error(f"Skipping {row['public_url']}: {e}")
