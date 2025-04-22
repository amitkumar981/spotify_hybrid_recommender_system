import pandas as pd
import logging
from data_cleaning import data_for_content_fitering
from content_based_filtering import apply_transformation, save_transformed_data

#configure logging
logger=logging.getLogger('transform_filtered_data')
logger.setLevel(logging.DEBUG)

#configure console handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add handler to logger
logger.addHandler(file_handler)

#configure formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Paths
filtered_data_path = "data/filtered_songs_data.csv"
save_path = "data/transformed_hybrid_data.npz"

def main():
    try:
        logging.info("Reading filtered data from CSV...")
        filtered_data = pd.read_csv(filtered_data_path)
        logging.info(f"Loaded {len(filtered_data)} records from {filtered_data_path}.")

        # Clean data for content filtering
        logging.info("Cleaning filtered data...")
        clean_filtered_data = data_for_content_fitering(filtered_data)
        logging.info(f"Data cleaned. Remaining records: {len(clean_filtered_data)}.")

        # Apply transformation
        logging.info("Applying content-based transformations...")
        transformed_data = apply_transformation(data=clean_filtered_data)
        logging.info("Transformation complete.")

        # Save transformed data
        logging.info(f"Saving transformed data to {save_path}...")
        save_transformed_data(transformed_data, save_path)
        logging.info("Data saved successfully.")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution.", exc_info=True)

if __name__ == "__main__":
    main()


