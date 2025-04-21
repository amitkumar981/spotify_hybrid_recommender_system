import streamlit as st
from content_based_filtering import content_recommendations
from collaborative_filtering import collaborative_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load

# Load cleaned data
songs_data = pd.read_csv("data/cleaned/cleaned_df.csv")


# Load transformed data for content-based filtering
transformed_data = load_npz("data/transformed_data.npz")

# Load track ids and filtered songs
track_ids = load("data/track_ids.npy", allow_pickle=True)
filtered_data = pd.read_csv("data/filtered_songs_data.csv")

# Load interaction matrix for collaborative filtering
sparse_matrix = load_npz("sparse_matrix.npz")

# UI
st.title('🎵 Spotify Song Recommender')
st.write('### Enter a song and artist, and get similar song recommendations 🎧')

# Input fields
song_name_input = st.text_input('Enter a song name:')
artist_name_input = st.text_input('Enter the artist name:')

# Normalize input for matching
song_name = song_name_input.strip().lower()
artist_name = artist_name_input.strip().lower()

# Number of recommendations
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

# Filtering type
filtering_type = st.selectbox('Select recommendation type:', ['Content-Based Filtering', 'Collaborative Filtering'])

# Button trigger
if st.button('Get Recommendations'):
    if filtering_type == 'Content-Based Filtering':
        match = (songs_data["name"].str.lower() == song_name) & (songs_data['artist'].str.lower() == artist_name)
        if match.any():
            st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
            recommendations = content_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                song_data=songs_data,
                transformed_data=transformed_data,
                k=k
            )
        else:
            st.warning(f"Sorry, we couldn't find **{song_name.title()}** by **{artist_name.title()}** in our database.")
            recommendations = None

    elif filtering_type == 'Collaborative Filtering':
        match = (filtered_data["name"].str.lower() == song_name) & (filtered_data['artist'].str.lower() == artist_name)
        if match.any():
            st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
            recommendations = collaborative_recommendation(
                song_name=song_name,
                track_ids=track_ids,
                artist_name=artist_name,
                songs_data=filtered_data,
                interaction_matrix=sparse_matrix
            )
        else:
            st.warning(f"Sorry, we couldn't find **{song_name.title()}** by **{artist_name.title()}** in our collaborative database.")
            recommendations = None

    # Display recommendations
    if recommendations is not None:
        for ind, recommendation in recommendations.iterrows():
            rec_song = recommendation['name'].title()
            rec_artist = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{rec_song}** by **{rec_artist}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            elif ind == 1:
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{rec_song}** by **{rec_artist}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{rec_song}** by **{rec_artist}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')

