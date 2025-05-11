import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import logging
import pandas as pd

logger = logging.getLogger(__name__)

#Note: It is customary to resize the image (makes the input size consistent) but we may not have to resize as much
def load_image_tensor_from_url(url: str) -> tf.Tensor:
    """
    Load an image from a URL and convert it to a TensorFlow tensor.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("L")
        image = image.resize((224, 224))
        return tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image from {url}: {e}")
        raise
    except (IOError, Image.UnidentifiedImageError) as e:
        logging.error(f"Error processing image from {url}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading image from {url}: {e}")
        raise

def csv_image_label_generator(csv_path: str, dataset: str = "train") -> tf.data.Dataset:
    """
    Generator yielding (image_tensor, label) pairs from a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset]
    for _, row in df.iterrows():
        try:
            image = load_image_tensor_from_url(row["public_url"])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, -1)  # (224, 224, 1)
            label = int(row["label_two"])
            yield image, label
        except Exception as e:
            logger.error(f"Skipping {row['public_url']}: {e}")
