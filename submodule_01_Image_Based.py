import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def load_and_preprocess_image(url_path=None, target_size=(224, 224)):
    try:
        # Load and preprocess the image from a URL
        response = requests.get(url_path, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        img = Image.open(response.raw).convert('RGB')
        
        # Resize and preprocess the image
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def extract_features(data, model, num_movies, TMDB_prefix="https://image.tmdb.org/t/p/w500"):
    movie_sequence = []
    features = []
    success_count = 0
    poster_col = data['poster_path']
    
    for row_idx in range(num_movies):
        if poster_col[row_idx] and poster_col[row_idx] != float('nan'):
            try:
                url = TMDB_prefix + poster_col[row_idx]
                preprocessed_image = load_and_preprocess_image(url_path=url)
            except Exception as e:
                print(f"Error {poster_col[row_idx]=}: {e}")
                
        else:
            preprocessed_image = None

        if preprocessed_image is not None:
            try:
                feature = model.predict(preprocessed_image)
                features.append(feature.flatten())
                success_count += 1  # Increment success counter
            except Exception as e:
                print(f"Error predicting features for index {row_idx}: {e}")
                features.append(None)
        else:
            features.append(None)
        
        movie_sequence.append(data['Title'])
    
    print(f"Successfully processed {success_count}/{num_movies} images.")
    return np.array(features), success_count, movie_sequence

def save_features(features, file_path):
    np.save(file_path, features)
    print(f"Features saved to {file_path}")



def load_features(file_path):
    if os.path.exists(file_path):
        print(f"Loading features from {file_path}")
        return np.load(file_path, allow_pickle=True)
    else:
        print(f"No saved features found at {file_path}. Please extract features first.")
        return None

def fetch_missing_poster_paths(metadata, api_key, poster_base_url="https://image.tmdb.org/t/p/original"):
    """
    Fetches missing poster paths for movies in the metadata and updates the dataset.

    Args:
    - metadata (pd.DataFrame): A DataFrame containing movie metadata with a 'poster_path' column.
    - api_key (str): The API key for accessing the external source (e.g., TMDB API).
    - poster_base_url (str): Base URL for movie poster paths (default: TMDB).

    Returns:
    - pd.DataFrame: Updated metadata with missing poster paths filled.
    """
 
    # Identify movies missing poster paths
    missing_posters = metadata[metadata['poster_path'].isnull()]
    print(f"Found {len(missing_posters)} movies with missing poster paths.")

    # Fetch missing poster paths using the API
    for index, row in missing_posters.iterrows():
        movie_id = row['imdb_id']
        response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                                params={"api_key": api_key})
        if response.status_code == 200:
            movie_data = response.json()
            poster_path = movie_data.get("poster_path")
            if poster_path:
                full_poster_path = f"{poster_base_url}{poster_path}"
                metadata.at[index, 'poster_path'] = full_poster_path
                print(f"Fetched poster path for movie_id {movie_id}: {full_poster_path}")
            else:
                print(f"No poster found for movie_id {movie_id}.")
        else:
            print(f"Failed to fetch data for movie_id {movie_id}. HTTP Status: {response.status_code}")

    return metadata

# Load existing features
def load_existing_features(file_path):
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"No existing features file found at {file_path}. Starting fresh.")
        return None

def get_movie_scores(movie_sequence, feature_path="./PosterFeatures/movie_features_merge_new.npy"):
    visual_embeddings = np.load(feature_path, allow_pickle=True)
    metadata = pd.read_csv("Datasets/Movies_Merged.csv")  # Contains 'movie_id', 'Genres', 'Description', etc.
    visual_embeddings = np.array([x[0] if x is not None else np.zeros_like(visual_embeddings[0][0]) for x in visual_embeddings])
    if len(visual_embeddings.shape) == 1:
        visual_embeddings = visual_embeddings.reshape(-1, 1)
    num_embeddings_rows = len(visual_embeddings)

    # print(f"{num_embeddings_rows=}, {len(metadata)=}, {visual_embeddings.shape=}")

    if num_embeddings_rows < len(metadata):
        padding_rows = len(metadata) - num_embeddings_rows
        zero_padding = np.zeros((padding_rows, visual_embeddings.shape[1]))
        visual_embeddings = np.vstack([visual_embeddings, zero_padding])

    # print(f"Padded visual_embeddings shape: {visual_embeddings.shape}")

    # Process content-based metadata
    metadata['Genres'] = metadata['Genres'].fillna("")  # Handle missing genres
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
    genre_features = vectorizer.fit_transform(metadata['Genres']).toarray()

    metadata['Description'] = metadata['Description'].fillna("")  # Handle missing descriptions
    tfidf = TfidfVectorizer(max_features=1000)
    description_features = tfidf.fit_transform(metadata['Description']).toarray()
    
    scaler = MinMaxScaler()
    visual_embeddings = np.nan_to_num(visual_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    visual_embeddings_scaled = scaler.fit_transform(visual_embeddings)

    genre_features_scaled = scaler.fit_transform(genre_features)
    description_features_scaled = scaler.fit_transform(description_features)

    # print(f"Shape of visual_embeddings_scaled: {visual_embeddings_scaled.shape}")
    # print(f"Shape of genre_features_scaled: {genre_features_scaled.shape}")
    # print(f"Shape of description_features_scaled: {description_features_scaled.shape}")

    # Ensure all feature arrays have matching lengths
    # min_length = min(
    #     visual_embeddings_scaled.shape[0], 
    #     genre_features_scaled.shape[0], 
    #     description_features_scaled.shape[0]
    # )
    # visual_embeddings_scaled = visual_embeddings_scaled[:min_length]
    # genre_features_scaled = genre_features_scaled[:min_length]
    # description_features_scaled = description_features_scaled[:min_length]
    # metadata = metadata.iloc[:min_length]

    # Combine features
    combined_features = np.hstack(
        (visual_embeddings_scaled, genre_features_scaled, description_features_scaled)
    )

    similarity_matrix = cosine_similarity(combined_features)

    # Aggregate similarity scores for multiple input movies
    def calculate_similarity_scores(titles, similarity_matrix):
        """
        Calculate similarity scores for all movies based on a sequence of input titles.

        Args:
        - titles (list of str): List of movie titles for similarity calculation.

        Returns:
        - np.array: Aggregated similarity scores for all movies in the dataset.
        """
        if not isinstance(titles, list):
            raise ValueError("Input titles must be a list of strings.")
        
        # Initialize an array to accumulate similarity scores
        scores = np.zeros(similarity_matrix.shape[0])

        for title in titles:
            if title not in metadata['title'].values:
                print(f"Warning: '{title}' not found in the metadata. Skipping.")
                continue
            
            movie_index = metadata[metadata['title'] == title].index[0]
            scores += similarity_matrix[movie_index]  # Add similarity scores for this movie
        
        scores = scores / np.sum(scores)
        return scores

    movie_scores = calculate_similarity_scores(movie_sequence, similarity_matrix)
    return movie_scores


if __name__ == "__main__":
    # data = pd.read_csv('Datasets/Movies_Merged.csv')
    # model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # # Usage example
    # features, success_count, movie_sequence = extract_features(data, model, len(data))
    # save_features(features, f"PosterFeatures/movie_features_merge_new.npy")
    # api_key = "7d54316704e8c29ee220d9b6484f6798"
    # features = load_features('PosterFeatures/movie_features_merge_new.npy')

    movie_sequence = ['The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring',
                      'The Lord of the Rings: The Two Towers', 'The Matrix', 'The Dark Knight Rises', 'Interstellar',
                      'Django Unchained', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight',
                      'The Hobbit: An Unexpected Journey', 'The Hobbit: The Desolation of Smaug', 
                      'The Hobbit: The Battle of the Five Armies']

    movie_scores = get_movie_scores(movie_sequence)

