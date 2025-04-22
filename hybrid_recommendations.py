import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommenderSystem:

    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based

    def __get_song_index(self, song_data, song_name, artist_name):
        match = song_data[
            (song_data["name"].str.lower() == song_name.lower()) &
            (song_data["artist"].str.lower() == artist_name.lower())
        ]
        if match.empty:
            raise ValueError("Song not found in song_data.")
        return match.index[0]

    def __calculate_content_based_similarities(self, song_index, transform_matrix):
        input_vector = transform_matrix[song_index].reshape(1, -1)
        return cosine_similarity(input_vector, transform_matrix)

    def __calculate_collaborative_based_similarities(self, track_id, track_ids, interaction_matrix):
        try:
            index = np.where(track_ids == track_id)[0][0]
        except IndexError:
            raise ValueError("Track ID not found in track_ids array.")
        input_vector = interaction_matrix[index]
        return cosine_similarity(input_vector, interaction_matrix)

    def __normalize_similarities(self, similarities_scores):
        max_val = np.max(similarities_scores)
        min_val = np.min(similarities_scores)
        if max_val == min_val:
            return np.zeros_like(similarities_scores)
        return (similarities_scores - min_val) / (max_val - min_val)

    def __weighted_combination(self, content_similarities_scores, collaborative_similarities_scores):
        return (
            self.weight_content_based * content_similarities_scores
            + self.weight_collaborative * collaborative_similarities_scores
        )

    def give_recommendations(self, song_name, artist_name, song_data, track_ids, interaction_matrix, transform_matrix):
        # Get song index and track ID
        song_index = self.__get_song_index(song_data, song_name, artist_name)
        input_track_id = song_data.loc[song_index, "track_id"]

        # Calculate similarity scores
        content_sim_scores = self.__calculate_content_based_similarities(song_index, transform_matrix)
        collab_sim_scores = self.__calculate_collaborative_based_similarities(input_track_id, track_ids, interaction_matrix)

        # Normalize scores
        norm_content = self.__normalize_similarities(content_sim_scores)
        norm_collab = self.__normalize_similarities(collab_sim_scores)

        # Combine scores
        weighted_scores = self.__weighted_combination(norm_content, norm_collab).ravel()

        # Exclude the input song itself
        weighted_scores[song_index] = -1

        # Get top indices
        top_indices = np.argsort(weighted_scores)[::-1][:self.number_of_recommendations]
        top_scores = weighted_scores[top_indices]
        top_track_ids = track_ids[top_indices]

        # Build result DataFrame
        scores_df = pd.DataFrame({"track_id": top_track_ids, "score": top_scores})
        top_songs = (
            song_data[song_data["track_id"].isin(top_track_ids)]
            .merge(scores_df, on="track_id")
            .sort_values(by="score", ascending=False)
            .drop_duplicates(subset=["track_id"])
            .drop(columns=["score"])
            .reset_index(drop=True)
        )

        return top_songs




    


