# -*- coding: utf-8 -*-
"""Spotify Content Based Filtering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NSqfgdkOL4Ztnp47-VIcTpJ0EPlRhElN
"""

!pip install kagglehub --upgrade

import kagglehub

# download the dataset from kaggle

kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = Path("/root/.cache/kagglehub/datasets/undefinenull/million-song-dataset-spotify-lastfm/versions/1")


songs_data_path = data_path / 'Music Info.csv'
users_data_path = data_path / 'User Listening History.csv'

# load the songs data

df_songs = pd.read_csv(songs_data_path)
df_songs.head()

"""# Getting the dataset ready"""

# shape of the data

df_songs.shape

# data info

df_songs.info()

# missing values

df_songs.isna().sum()

# ratio of missing values in data

(
    df_songs
    .isna()
    .mean()
    .sort_values(ascending=False)
    .head(2)
    .mul(100)
)

# duplicates in the data based on spotify_id

(
    df_songs
    .duplicated(subset="spotify_id")
    .sum()
)

# drop duplicates

df_songs.drop_duplicates(subset=["spotify_id","year","duration_ms"],inplace=True)

# check for duplicates

(
    df_songs
    .duplicated(subset=["spotify_id","year","duration_ms"])
    .sum()
)

df_songs.head()

# reset the index

df_songs.reset_index(drop=True,inplace=True)

# remove columns not required for collaborative filtering

cols_to_remove = ["track_id","name","spotify_preview_url","spotify_id","genre"]


df_colab_filtering = df_songs.drop(columns=cols_to_remove)

df_colab_filtering

# check for missing values

df_colab_filtering.isna().sum()

# fill the tags column missing values with string "no_tags"

df_colab_filtering.fillna({"tags":"no_tags"},inplace=True)

# check for missing values

df_colab_filtering.isna().sum().sum()

# artist names as lower case

df_colab_filtering["artist"] = df_colab_filtering["artist"].str.lower()

# number of unique artists

(
    df_songs
    .loc[:,'artist']
    .nunique()
)

# number of unique year values

(
    df_songs
    .loc[:,'year']
    .nunique()
)

# min and max values of numerical columns

(
    df_songs
    .select_dtypes(include='number')
    .agg(['min','max'])
)

# value counts for the tags

(
    df_songs
    .loc[:,'tags']
    .str.lower()
    .str.split(',')
    .explode()
    .str.strip()
    .value_counts()
)

# value counts for the tags

(
    df_songs
    .loc[:,'tags']
    .str.lower()
    .str.split(',')
    .explode()
    .str.strip()
    .value_counts()
    .loc[lambda ser: ser >= 1000]
)

! pip install category_encoders

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

df_colab_filtering.head(3)

df_colab_filtering.shape

# cols to transform

frequency_encode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]

len(frequency_encode_cols + ohe_cols + standard_scale_cols + min_max_scale_cols)

# transform the data

transformer = ColumnTransformer(transformers=[
    ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_encode_cols),
    ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
    ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
    ("standard_scale", StandardScaler(), standard_scale_cols),
    ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

transformer

# fit the transformer

transformer.fit(df_colab_filtering)

# transform the data

transformed_df = transformer.transform(df_colab_filtering)

transformed_df.shape

transformed_df

from sklearn.metrics.pairwise import cosine_similarity

# fetch a song where artist is Shakira

df_songs.loc[df_songs["artist"] == "Shakira"]

# build input vector

df_songs[df_songs["name"] == "Whenever, Wherever"]

song_input = df_colab_filtering[df_songs["name"] == "Whenever, Wherever"]

song_input

# input vector to calculate similarity

input_vector = transformer.transform(song_input)

input_vector

# calculate the similarity matrix

similarity_scores = cosine_similarity(transformed_df,input_vector)

similarity_scores.shape

similarity_scores

top_10_songs_indexes = np.argsort(similarity_scores.ravel())[-11:][::-1]

top_10_songs_indexes

top_10_songs_names = df_songs.iloc[top_10_songs_indexes]

top_10_songs_names

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

# recommend song using the function

recommend("Whenever, Wherever",songs_data=df_songs,transformed_data=transformed_df,k=10)