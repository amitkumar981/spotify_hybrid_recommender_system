import streamlit as st
from content_based_filtering import content_recommendations
from scipy.sparse import load_npz
import pandas as pd

# Load cleaned data
cleaned_data_path = "data/cleaned/cleaned_df.csv"
songs_data = pd.read_csv(cleaned_data_path)

# Load transformed data
transformed_data_path = "data/transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

# Title
st.title('🎵 Spotify Song Recommender')

# Subheader
st.write('### Enter a song and artist, and get similar song recommendations 🎧')

# Input fields
song_name_input = st.text_input('Enter a song name:')
artist_name_input = st.text_input('Enter the artist name:')

# Normalize for matching
song_name = song_name_input.lower()
artist_name = artist_name_input.lower()

# Number of recommendations
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

# Submit
if st.button('Get Recommendations'):
    if ((songs_data["name"].str.lower() == song_name) & 
        (songs_data['artist'].str.lower() == artist_name)).any():

        st.write(f'Recommendations for **{song_name_input}** by **{artist_name_input}**')
        recommendations = content_recommendations(
            song_name=song_name,
            artist_name=artist_name,
            song_data=songs_data,
            transformed_data=transformed_data,
            k=k
        )

        for ind, recommendation in recommendations.iterrows():
            rec_song_name = recommendation['name'].title()
            rec_artist_name = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
            elif ind == 1:
                st.markdown("### Next Up 🎵")
            else:
                st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")

            st.audio(recommendation['spotify_preview_url'])
            st.write('---')

    else:
        st.warning(f"Sorry, we couldn't find **{song_name_input}** by **{artist_name_input}** in our database.")

