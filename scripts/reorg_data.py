import os
import random
import logging
from google.cloud import storage

SOURCE_BUCKET_NAME = "dsci471"
DEST_DATASET_PREFIX = "reorg_80_10_10/"
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_EXTENSIONS = ('.jpg', '.jpeg')

def process_and_split_dataset_gcs():
    try:
        storage_client = storage.Client()
        source_bucket = storage_client.bucket(SOURCE_BUCKET_NAME)
        logging.info(f"Successfully connected to GCS. Source: gs://{SOURCE_BUCKET_NAME}")
    except Exception as e:
        logging.error(f"Failed to connect to GCS or buckets: {e}")
        return

    all_files_by_final_category = {
        "NORMAL": [],
        "PNEUMONIA_VIRAL": [],
        "PNEUMONIA_BACTERIAL": []
    }

    # Iterate through the original Kaggle splits ('train', 'val', 'test')
    # to collect all files first.
    original_splits = ['train', 'val', 'test']
    original_top_categories = ['NORMAL', 'PNEUMONIA']

    logging.info("Starting scan of source GCS location...")
    for orig_split in original_splits:
        for orig_cat in original_top_categories:
            current_source_prefix = f"{orig_split}/{orig_cat}/"
            logging.info(f"Scanning prefix: gs://{SOURCE_BUCKET_NAME}/{current_source_prefix}")
            
            blobs = list(source_bucket.list_blobs(prefix=current_source_prefix))
            if not blobs:
                logging.warning(f"No blobs found in gs://{SOURCE_BUCKET_NAME}/{current_source_prefix}")
                continue

            for blob in blobs:
                blob_name_lower = blob.name.lower()
                if not blob_name_lower.endswith(IMAGE_EXTENSIONS):
                    if blob_name_lower != current_source_prefix:
                         logging.debug(f"Skipping non-image file or directory marker: {blob.name}")
                    continue

                filename_lower = blob.name.split('/')[-1].lower()

                if orig_cat == 'NORMAL':
                    all_files_by_final_category["NORMAL"].append(blob)
                elif orig_cat == 'PNEUMONIA':
                    is_viral_keyword = "virus" in filename_lower
                    is_bacterial_keyword = "bacteria" in filename_lower

                    if is_viral_keyword:
                        all_files_by_final_category["PNEUMONIA_VIRAL"].append(blob)
                    elif is_bacterial_keyword:
                        all_files_by_final_category["PNEUMONIA_BACTERIAL"].append(blob)
                    else:
                        logging.debug(f"Pneumonia file {blob.name} not categorized as VIRAL or BACTERIAL based on keywords. It will be excluded from these subcategories.")
    
    logging.info("Finished scanning source files. Summary of files collected:")
    for category, files in all_files_by_final_category.items():
        logging.info(f"Category '{category}': {len(files)} files")

    # Shuffle, split, and copy files for each final category
    logging.info("Starting shuffling, splitting, and copying files to destination...")
    for final_category, blob_list in all_files_by_final_category.items():
        if not blob_list:
            logging.info(f"No files to process for category: {final_category}. Skipping.")
            continue

        random.shuffle(blob_list)
        

        total_files = len(blob_list)
        train_val_split_point = int(total_files * TRAIN_SPLIT_RATIO)
        val_test_split_point = int(total_files * (TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO))

        train_blobs = blob_list[:train_val_split_point]
        val_blobs = blob_list[train_val_split_point:val_test_split_point]
        test_blobs = blob_list[val_test_split_point:]

        logging.info(f"Category '{final_category}': {len(train_blobs)} train files, {len(val_blobs)} validation files, {len(test_blobs)} test files.")

        # Process training files
        for blob in train_blobs:
            original_filename = blob.name.split('/')[-1]
            destination_blob_name = f"{DEST_DATASET_PREFIX.rstrip('/')}/train/{final_category}/{original_filename}"
            try:
                source_bucket.copy_blob(blob, source_bucket, destination_blob_name)
                logging.debug(f"Copied TRAIN: {blob.name} to gs://{SOURCE_BUCKET_NAME}/{destination_blob_name}")
            except Exception as e:
                logging.error(f"Failed to copy {blob.name} to {destination_blob_name}: {e}")

        # Process validation files
        for blob in val_blobs:
            original_filename = blob.name.split('/')[-1]
            destination_blob_name = f"{DEST_DATASET_PREFIX.rstrip('/')}/val/{final_category}/{original_filename}"
            try:
                source_bucket.copy_blob(blob, source_bucket, destination_blob_name)
                logging.debug(f"Copied VALIDATION: {blob.name} to gs://{SOURCE_BUCKET_NAME}/{destination_blob_name}")
            except Exception as e:
                logging.error(f"Failed to copy {blob.name} to {destination_blob_name}: {e}")
        
        # Process testing files
        for blob in test_blobs:
            original_filename = blob.name.split('/')[-1]
            destination_blob_name = f"{DEST_DATASET_PREFIX.rstrip('/')}/test/{final_category}/{original_filename}"
            try:
                source_bucket.copy_blob(blob, source_bucket, destination_blob_name)
                logging.debug(f"Copied TEST: {blob.name} to gs://{SOURCE_BUCKET_NAME}/{destination_blob_name}")
            except Exception as e:
                logging.error(f"Failed to copy {blob.name} to {destination_blob_name}: {e}")
        
        logging.info(f"Finished processing category: {final_category}")

    logging.info("Dataset reorganization and splitting complete.")

if __name__ == '__main__':
    process_and_split_dataset_gcs()
