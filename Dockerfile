# Set up the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app/

# Copy the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the sparse matrix separately into /app/
COPY sparse_matrix.npz .

# Copy all other data files into /app/data/
COPY ./data/filtered_songs_data.csv \
     ./data/track_ids.npy \
     ./data/cleaned/cleaned_df.csv \
     ./data/transformed_data.npz \
     ./data/transformed_hybrid_data.npz \
     ./data/ data/

# Copy all Python scripts into /app/
COPY app.py \
     collaborative_filtering.py \
     content_based_filtering.py \
     hybrid_recommendations.py \
     data_cleaning.py \
     transform_filtered_data.py \
     ./

# Expose Streamlit's default port
EXPOSE 8000

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
