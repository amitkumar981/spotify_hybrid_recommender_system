#importing libraries
import numpy as np
import pandas as pd
import os
import logging

#configure logging
logger=logging.getLogger('data_cleaning')
logger.setLevel(logging.DEBUG)

#configure console handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add handler to logger
logger.addHandler(file_handler)

#configure formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Function to load data
def load_data(data_path: str):
    """
    Load data from a CSV file.
    
    Parameters:
    - data_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None


# Function to perform data cleaning
def perform_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input data.

    Steps:
    - Remove duplicate rows based on 'track_id'.
    - Drop unnecessary columns ('genre', 'spotify_id').
    - Fill missing 'tags' with 'no tags'.
    - Convert 'name', 'artist', and 'tags' to lowercase.

    Parameters:
    - data (pd.DataFrame): Raw input data.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    try:
        logger.info('Initializing data cleaning...')
        cleaned_data = (
            data
            .drop_duplicates(subset='track_id')  # Remove duplicates
            .drop(columns=['genre', 'spotify_id'], errors='ignore')  # Drop unwanted columns
            .fillna({'tags': 'no tags'})  # Fill missing tags
            .assign(
                name=lambda x: x['name'].str.lower(),  # Convert 'name' to lowercase
                artist=lambda x: x['artist'].str.lower(),  # Convert 'artist' to lowercase
                tags=lambda x: x['tags'].str.lower()  # Convert 'tags' to lowercase
            )
            .reset_index(drop=True)  # Reset index
        )
        logger.info('Data cleaning completed successfully.')
        return cleaned_data
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        return cleaned_data
    
def data_for_content_fitering(cleaned_data):
    return(
        cleaned_data.drop(columns=["track_id","name","spotify_preview_url"])
    )

def save_data(data: pd.DataFrame, save_data_path: str):
    try:
        ensure_directory_exists(os.path.dirname(save_data_path))
        data.to_csv(save_data_path, index=False)
        logger.info(f"Data saved to {save_data_path}")
    except Exception as e:
        logger.error(f"Error saving data to {save_data_path}: {e}")

def ensure_directory_exists(directory: str):
    """Ensure a directory exists; if not, create it."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error ensuring directory exists {directory}: {e}")
        raise

def get_root_directory() -> str:
    """Get the root directory (one level up from this script's location)."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, ''))  
        logger.debug(f"Current directory: {current_dir}")
        logger.debug(f"Resolved root directory: {root_dir}")
        return root_dir
    except Exception as e:
        logger.error('Error getting root directory: %s', e)
        return None

def main():

    #get root directory
    root_path=get_root_directory()

    data_path=os.path.join(root_path,'data','raw','Music info.csv')

    data=load_data(data_path)

    #perform data cleaning
    cleaned_data=perform_data_cleaning(data)

    save_path = os.path.join(root_path, 'data', 'cleaned', 'cleaned_df.csv')

    save_data(cleaned_data,save_path)

if __name__ == "__main__":
    main()



    
    

    



    
    
