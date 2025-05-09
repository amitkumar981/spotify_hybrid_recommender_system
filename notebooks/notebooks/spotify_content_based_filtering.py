# -*- coding: utf-8 -*-
"""spotify_content_based_filtering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g22qsfBCEuGa60puKofkAAnEySfKDRKn
"""

! pip install kagglehub --upgrade

#download dataset from kaggle
import kagglehub

kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")

!pip install missingno

# import libraries
import numpy as np
import pandas as pd
from pathlib import Path
import missingno as msno

data_path = Path('/root/.cache/kagglehub/datasets/undefinenull/million-song-dataset-spotify-lastfm/versions/1')

songs_data_path = data_path / 'music info.csv'
users_data_path = data_path / 'User Listening History.csv'

data_path = Path('/root/.cache/kagglehub/datasets/undefinenull/million-song-dataset-spotify-lastfm/versions/1')

# List all files in the directory
for file in data_path.iterdir():
    print(file)

from pathlib import Path

data_path = Path('/root/.cache/kagglehub/datasets/undefinenull/million-song-dataset-spotify-lastfm/versions/1')

songs_data_path= data_path /'Music Info.csv'
users_data = data_path /'User Listening History.csv'

songs_df=pd.read_csv(songs_data_path)
songs_df.head()

songs_df.shape

#checkinng missing values from dataset
songs_df.isnull().sum()

#plot missing values'
msno.matrix(songs_df)

#ratio of missing values
songs_df.isnull().mean().sort_values(ascending=False).head(2)*100

duplicate_index=songs_df[songs_df.duplicated(subset='spotify_id',keep=False)].sort_values(by='spotify_id').index.to_list()
print(len(duplicate_index))

#drop duplicate index from data set
cleaned_df=songs_df.drop(index=duplicate_index)

#check again duplicates
cleaned_df.duplicated(subset=['spotify_id','year']).sum()

cleaned_df.reset_index(drop=True,inplace=True)

cleaned_df.columns

cols_to_remove=['track_id','name','spotify_preview_url', 'spotify_id','genre']
cleaned_df.drop(columns=cols_to_remove,inplace=True)
cleaned_df.shape

#cheking missing values
cleaned_df.isnull().sum()

#fill missing values
cleaned_df.fillna({'tags':'no_tags'},inplace=True)

#check missing values
cleaned_df.isnull().sum()

cleaned_df['artist'].str.lower().nunique()

cleaned_df['year'].nunique()

#min and max values in num colums
cleaned_df.select_dtypes(include=['number']).agg(['min','max'])

cleaned_df['tags'].str.lower().str.split(',').explode().value_counts()

cleaned_df['tags'].str.lower().str.split(',').explode().value_counts().loc[lambda ser: ser >=1000]

! pip install category_encoders

from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

songs_df.head()

frequency_encode_cols=['year']
ohe_encode_cols=['artist','time_signature','key']
tfidf_encode_cols='tags'
standard_scale_cols=['duration_ms','loudness','tempo']
min_max_scale_cols=['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']

len(frequency_encode_cols+ohe_encode_cols+standard_scale_cols+min_max_scale_cols)

transformer=ColumnTransformer(transformers=[
    ('frequency_encode',CountEncoder(normalize=True,return_df=True),frequency_encode_cols),
    ('ohe_encode',OneHotEncoder(handle_unknown='ignore'),ohe_encode_cols),
    ('tfidf_encode',TfidfVectorizer(max_features=90),tfidf_encode_cols),
    ('standard_scaler',StandardScaler(),standard_scale_cols),
    ('min max scaler',MinMaxScaler(),min_max_scale_cols)
],remainder='passthrough',n_jobs=-1)
transformer

transformed_df=transformer.fit_transform(cleaned_df)

transformed_df.shape

transformed_df

songs_df.sample(5)

songs_df[songs_df['artist']=="Stereo MC's"]

song_input=cleaned_df[songs_df['name']=='Step It Up']

input_vector=transformer.transform(song_input)

input_vector.shape

from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(input_vector,transformed_df)

similarity_score

top_10_song_indexs=np.argsort(similarity_score).ravel()[-11:][::-1]

top_10_song_indexs

songs_df.iloc[top_10_song_indexs]

def recommend(song_name, songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    songs_data (DataFrame): The DataFrame containing song information.
    transformed_data (ndarray): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    # filter out the song from data
    song_row = songs_data.loc[songs_data["name"] == song_name,:]
    if song_row.empty:
        print("Song not found in the dataset.")
    else:
        # get the index of song
        song_index = song_row.index[0]
        print(song_index)
        # generate the input vector
        input_vector = transformed_data[song_index].reshape(1,-1)
        # calculate similarity scores
        similarity_scores = cosine_similarity(input_vector, transformed_data)
        print(similarity_scores.shape)
        # get the top k songs
        top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
        print(top_k_songs_indexes)
        # get the top k songs names
        top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
        # print the top k songs
        top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
        return top_k_list

recommend("Whenever, Wherever",songs_data=songs_df,transformed_data=transformed_df,k=10)

