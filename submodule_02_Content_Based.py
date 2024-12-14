import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

#Content Based Filtering
# Enhance combined features with weights
#Weights: Genre = 2, Keywords = 3, Summary, Director = 1
def combine_features_with_weights(row, genre_weight=2, keyword_weight=3):
    description = row['Description'] if not pd.isnull(row['Description']) else ''
    genres = (row['Genres'] + ' ') * genre_weight if not pd.isnull(row['Genres']) else ''
    keywords = (row['keywords'] + ' ') * keyword_weight if not pd.isnull(row['keywords']) else ''
    tagline = row['tagline'] if not pd.isnull(row['tagline']) else ''
    written_by = row['Written by'] if not pd.isnull(row['Written by']) else ''
    directed_by = row['Directed by'] if not pd.isnull(row['Directed by']) else ''
    
    return f"{description} {genres} {keywords} {tagline} {written_by} {directed_by}"

#Function can receive any number of inputs 
def get_movie_scores(movie_titles, path):
    # Load the dataset
    movies_data = pd.read_csv(path)
    movies_data['combined_features'] = movies_data.apply(combine_features_with_weights, axis=1)

    # Vectorize the combined features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['combined_features'])

    # Ensure movie_titles is a list
    if isinstance(movie_titles, str):
        movie_titles = [movie_titles]
    
    # Find indices of the input movies
    indices = pd.Series(movies_data.index, index=movies_data['Title']).drop_duplicates()
    input_indices = [indices.get(title) for title in movie_titles if title in indices]
    
    if not input_indices:
        raise ValueError("None of the provided movies are found in the dataset.")
    
    # Aggregate similarity scores for all provided movies
    aggregated_scores = sum(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten() for idx in input_indices)

    # Normalize the aggregated scores
    aggregated_scores = aggregated_scores / (sum(aggregated_scores) + 1e-6)

    return aggregated_scores

# example_movie = ['The Godfather', 'The Traitor']
# recommendations = get_recommendations_multi_content_based(example_movie)
# print(recommendations)

# example_movie = ['The Martian', 'Red Planet']
# recommendations = get_recommendations_multi_content_based(example_movie)
# print(recommendations)