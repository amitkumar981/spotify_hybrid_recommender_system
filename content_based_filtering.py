# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from data_cleaning import data_for_content_fitering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import os
import pickle
import logging

#configure logging
logger=logging.getLogger('content_based_filtering')
logger.setLevel(logging.DEBUG)

#configure console handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add handler to logger
logger.addHandler(file_handler)

#configure formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

cleaned_data_path=('.', 'data', 'cleaned', 'cleaned_df.csv')

# Load data
def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None

# Train and save transformer
def train_transformer(data: pd.DataFrame):
    try:
        logger.info("Initializing transformer training...")

        frequency_encode_cols = ['year']
        ohe_encode_cols = ['artist', 'time_signature', 'key']
        tfidf_encode_col = 'tags'
        standard_scale_cols = ['duration_ms', 'loudness', 'tempo']
        min_max_scale_cols = ['danceability', 'energy', 'speechiness', 'acousticness',
                              'instrumentalness', 'liveness', 'valence']

        transformer = ColumnTransformer(transformers=[
            ('frequency_encode', CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
            ('ohe_encode', OneHotEncoder(handle_unknown='ignore'), ohe_encode_cols),
            ('tfidf_encode', TfidfVectorizer(max_features=85), tfidf_encode_col),
            ('standard_scaler', StandardScaler(), standard_scale_cols),
            ('min_max_scaler', MinMaxScaler(), min_max_scale_cols)
        ], remainder='passthrough', n_jobs=-1)

        transformer.fit(data)
        with open(os.path.join('.','transformer.pkl'), 'wb') as f:
            pickle.dump(transformer, f)

        logger.info("Transformer trained and saved.")
    except Exception as e:
        logger.error(f"Error during transformer training: {e}")

# Load transformer and transform data
def apply_transformation(data):
    try:
        logger.info("Loading transformer and transforming data...")
        with open(os.path.join('.','transformer.pkl'), 'rb') as f:
            transformer = pickle.load(f)
        transformed = transformer.transform(data)
        logger.info("Data transformed successfully.")
        return transformed
    except Exception as e:
        logger.error(f"Error applying transformation: {e}")
        return None

# Save transformed sparse matrix
def save_transformed_data(transformed_data, save_path):
    try:
        save_npz(save_path, transformed_data)
        logger.info(f"Transformed data saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving transformed data: {e}")

# Compute cosine similarity
def calculate_cosine_similarity(input_vector, data):
    return cosine_similarity(input_vector, data)

# Recommend similar songs
def content_recommendations(song_name, artist_name, song_data, transformed_data, k=10):
    try:
        song_name = song_name.lower()
        artist_name = artist_name.lower()
        song_row = song_data.loc[
            (song_data['name'].str.lower() == song_name) & 
            (song_data['artist'].str.lower() == artist_name)
        ]
        if song_row.empty:
            logger.warning("Song not found in dataset.")
            return pd.DataFrame()

        song_index = song_row.index[0]
        input_vector = transformed_data[song_index]

        similarity_scores = cosine_similarity(input_vector, transformed_data)
        top_k_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

        recommended = song_data.iloc[top_k_indexes][['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
        return recommended
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

# Main function
def main():
    data_path = os.path.join('.', 'data', 'cleaned', 'cleaned_df.csv')
    
    df = load_data(data_path)
    if df is None:
        return
    filtered_data=data_for_content_fitering(df)
    
    #train transformer
    train_transformer(filtered_data)
    
    
    #apply transformation
    transformed = apply_transformation(filtered_data)
    if transformed is not None:
        save_path = os.path.join('.', 'data', 'transformed_data.npz')
        save_transformed_data(transformed, save_path)

if __name__ == "__main__":
    main()














    
    


    





    



