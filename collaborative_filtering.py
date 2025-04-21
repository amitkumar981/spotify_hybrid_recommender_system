# Import libraries
import numpy as np
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import os

# Configure logging
logger = logging.getLogger('collaborative_filtering')
logger.setLevel(logging.DEBUG)
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load music info data
def load_songs_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading songs data from: {data_path}")
        songs_df = pd.read_csv(data_path)
        logger.info("Songs data loaded successfully.")
        return songs_df
    except Exception as e:
        logger.error(f"Error loading songs data from {data_path}: {e}")
        raise

# Load user history from data
def load_user_data(user_info_path: str) -> dd.DataFrame:
    try:
        logger.info(f"Loading user data from: {user_info_path}")
        user_df = dd.read_csv(user_info_path)
        logger.info("User data loaded successfully.")
        return user_df
    except Exception as e:
        logger.error(f"Error loading user data from {user_info_path}: {e}")
        raise

def filtered_songs_data(songs_df: pd.DataFrame, track_ids: list) -> pd.DataFrame:
    logger.info("Filtering songs data based on provided track IDs.")
    filtered_data = songs_df[songs_df['track_id'].isin(track_ids)]
    filtered_data.sort_values(by="track_id", inplace=True)
    filtered_data.reset_index(drop=True, inplace=True)
    logger.info(f"Filtered data contains {len(filtered_data)} rows.")
    return filtered_data

def save_data(data: pd.DataFrame, save_data_path: str) -> None:
    try:
        logger.info(f"Saving DataFrame to: {save_data_path}")
        data.to_csv(save_data_path, index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save data to {save_data_path}: {e}")
        raise

def create_interaction_matrix(data, track_ids: list) -> csr_matrix:
    try:
        logger.info("Creating interaction matrix.")
        data['playcount'] = data['playcount'].astype(np.float64)
        data = data.categorize(columns=['track_id', 'user_id'])
        
        user_mapping=data['user_id'].cat.codes
        track_mapping=data['track_id'].cat.codes

         # get the list of track_ids
        track_ids = data['track_id'].cat.categories.values
        track_ids_save_path = "data/track_ids.npy"
    
        # save the categories
        np.save(track_ids_save_path, track_ids, allow_pickle=True)

        data=data.assign(user_id_idx=user_mapping,track_id_idx=track_mapping)
        

        interaction_array = data.groupby(['track_id_idx', 'user_id_idx'])['playcount'].sum().reset_index().compute()

        row_indices = interaction_array['track_id_idx']
        col_indices = interaction_array['user_id_idx']
        values = interaction_array['playcount']

        n_tracks=row_indices.nunique()
        n_users=col_indices.nunique()
      

        interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
        logger.info("Sparse interaction matrix created successfully.")

        return interaction_matrix
    except Exception as e:
        logger.error(f"Error creating interaction matrix: {e}")
        raise

def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    try:
        logger.info(f"Saving sparse matrix to {file_path}")
        save_npz(file_path, matrix)
        logger.info("Sparse matrix saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save sparse matrix to {file_path}: {e}")
        raise

def collaborative_recommendations(song_name, user_data, song_data, interaction_matrix, k=5):
    song_row = song_data[song_data['name'] == song_name]

    if song_row.empty:
        print(f"Song '{song_name}' not found.")
        return pd.DataFrame()

    input_track_id = song_row['track_id'].values[0]

    if not pd.api.types.is_categorical_dtype(user_data['track_id']):
        print("track_id must be categorical.")
        return pd.DataFrame()

    try:
        track_idx = user_data['track_id'].cat.categories.get_loc(input_track_id)
    except KeyError:
        print(f"Track ID {input_track_id} not in user data categories.")
        return pd.DataFrame()

    input_array = interaction_matrix[track_idx]
    similarity_scores = cosine_similarity(input_array, interaction_matrix).flatten()
    top_k_idx = np.argsort(similarity_scores)[::-1][1:k+1]
    recommended_track_ids = user_data['track_id'].cat.categories[top_k_idx]
    top_scores = similarity_scores[top_k_idx]

    temp_df = pd.DataFrame({"track_id": recommended_track_ids, "score": top_scores})

    top_k_songs = (
        song_data[song_data["track_id"].isin(recommended_track_ids)]
        .merge(temp_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )

    return top_k_songs

def main():
    user_info_path = os.path.join('.', 'data', 'raw', 'User Listening History.csv')
    songs_info_path = os.path.join('.', 'data', 'cleaned', 'cleaned_df.csv')

    songs_df = load_songs_data(songs_info_path)
    user_df = load_user_data(user_info_path)

    track_ids=user_df.loc[:,'track_id'].unique().compute().tolist()
    filtered_df = filtered_songs_data(songs_df, track_ids)

    save_data(filtered_df, os.path.join('.', 'data', 'filtered_songs_data.csv'))
    sparse_matrix = create_interaction_matrix(user_df, track_ids)

    save_sparse_matrix(sparse_matrix, os.path.join('.', 'sparse_matrix.npz'))

if __name__ == "__main__":
    main()

















    







