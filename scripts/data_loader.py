import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import io
import logging
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# BASE_IMAGE_FOLDER = "/mnt/c/Users/NesFa/Desktop/trainingFiles"


#Note: It is customary to resize the image (makes the input size consistent) but we may not have to resize as much
def load_image_tensor_from_url(url: str, image_size: tuple[int, int] = (1348, 987)) -> tf.Tensor:
    """
    Load an image from a URL and convert it to a TensorFlow tensor.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("L")
        image = image.resize(image_size)
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

def load_image_tensor_from_file(file_path: str, image_size: tuple[int, int] = (1348, 987)) -> tf.Tensor:
    """
    Load an image from a local file path.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found at path: {file_path}")

        # Open the image file using PIL
        image = Image.open(file_path)
        
        # Convert to grayscale ("L" mode)
        image_gray = image.convert("L")
        
        # Resize the image
        # PIL's resize method expects (width, height)
        image_resized = image_gray.resize(image_size) 
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image_resized)
        
        # Convert the NumPy array to a TensorFlow tensor
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
        
        # logging.debug(f"Successfully loaded and processed image from {file_path}")
        return image_tensor

    except FileNotFoundError as fnf_error:
        logging.error(f"FileNotFoundError for image at {file_path}: {fnf_error}")
        raise # Re-raise the specific error to be caught by the caller if needed
    except UnidentifiedImageError as unident_error:
        logging.error(f"PIL UnidentifiedImageError for image at {file_path}: {unident_error}. The file may be corrupted or not a valid image format.")
        raise
    except IOError as io_error: # Catches general I/O errors from PIL processing
        logging.error(f"IOError processing image at {file_path}: {io_error}")
        raise
    except Exception as e: # Catch-all for any other unexpected errors
        logging.error(f"Unexpected error loading image from file {file_path}: {e.__class__.__name__} - {e}", exc_info=False) # exc_info=False to keep log cleaner
        raise

def csv_image_label_generator(csv_path: str, dataset: str = "train", binary_class: bool = False, shuffle: bool = True, to_rgb: bool = False, image_size: tuple[int, int] = (1348, 987)) -> tf.data.Dataset:
    """
    Generator yielding (image_tensor, label) pairs from a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset]
    num_classes = df["label_two"].nunique()
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    for _, row in df.iterrows():
        try:
            # public_url = row["public_url"]
            # relative_path_start_index = public_url.index("dsci471/") + len("dsci471/")
            # relative_path = public_url[relative_path_start_index:]
            # local_file_path = os.path.join(BASE_IMAGE_FOLDER, relative_path)
            # image = load_image_tensor_from_file(local_file_path, image_size=image_size)
            image = load_image_tensor_from_url(row["public_url"], image_size=image_size)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, -1)  # (224, 224, 1)
            label = int(row["label_two"])

            if binary_class:
                if label != 0:
                    label = 1
            else:
                label = tf.one_hot(label, depth=num_classes, dtype=tf.int32)

            if to_rgb:
                image = tf.image.grayscale_to_rgb(image)
            
            if dataset == "val":
                logger.info(f"VAL {_}")

            yield image, label
        except Exception as e:
            logger.error(f"Skipping {row['public_url']}: {e}")
