import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import networkx as nx
import numpy as np


# Preprocess and normalize numerical features
def preprocess_numerical_features(movies_data):
    scaler = StandardScaler()
    numerical_features = movies_data[["Release Year", "runtime"]].fillna(0)
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    return scaled_numerical_features


# Combine text and numerical features
def combine_features_with_weights(row, weights=(1, 2, 3, 1, 1, 1, 1, 1)):
    (
        description_weight,
        genre_weight,
        keyword_weight,
        tagline_weight,
        writer_weight,
        director_weight,
        language_weight,
        company_weight,
    ) = weights
    
    description = (row["Description"] * description_weight if not pd.isnull(row["Description"]) else "")
    genres = ((row["Genres"] + " ") * genre_weight if not pd.isnull(row["Genres"]) else "")
    keywords = ((row["keywords"] + " ") * keyword_weight if not pd.isnull(row["keywords"]) else "")
    tagline = row["tagline"] * tagline_weight if not pd.isnull(row["tagline"]) else ""
    written_by = (row["Written by"] * writer_weight if not pd.isnull(row["Written by"]) else "")
    directed_by = (row["Directed by"] * director_weight if not pd.isnull(row["Directed by"]) else "")
    language = (row["original_language"] * language_weight if not pd.isnull(row["original_language"]) else "")
    production_companies = ((row["production_companies"] + " ") * company_weight 
                            if not pd.isnull(row["production_companies"]) else "")

    text_features = f"{description} {genres} {keywords} {tagline} {written_by} {directed_by} {language} {production_companies}"
    return f"{text_features}"


def build_graph_bm25_numeric(movies_data, bm25, scaled_numerical_features, numeric_weight=10, threshold=0.2):
    num_movies = len(bm25.get_scores([])) 
    G = nx.Graph()

    for i in range(num_movies):
        query = movies_data.loc[i, "combined_features"].split()
        bm25_scores = bm25.get_scores(query)
        movie_numerical_features = scaled_numerical_features[i]
        numerical_similarity = cosine_similarity(
            [movie_numerical_features], scaled_numerical_features).flatten()

        for j in range(i+1, num_movies):
            if i != j:  
                weight = bm25_scores[j] + numerical_similarity[j] * numeric_weight
                if weight > threshold: 
                    G.add_edge(i, j, weight=weight)

    return G

def build_graph_tfidf_numeric(sim, scaled_numerical_features, numeric_weight=0.1, threshold=0.25):
    num_movies = sim.shape[0] 
    G = nx.Graph()

    for i in range(num_movies):
        movie_numerical_features = scaled_numerical_features[i]
        numerical_similarity = cosine_similarity(
            [movie_numerical_features], scaled_numerical_features).flatten()

        for j in range(i+1, num_movies):
            if i != j:  
                weight = (sim[i,j] + numerical_similarity[j] * numeric_weight) / (1 + numeric_weight)
                if weight > threshold: 
                    G.add_edge(i, j, weight=weight)

    return G


def personalized_pagerank_multiple(movies_data, graph, start_nodes, alpha=0.15):
    # Initialize personalization vector
    personalization = {node: 0 for node in graph.nodes}
    for node in start_nodes:
        personalization[node] = 1 / len(start_nodes)  # Evenly distribute scores

    # Compute PageRank
    pagerank_scores = nx.pagerank(graph, alpha=alpha, personalization=personalization)
   
    # Ensure pagerank_scores is sorted according to movie indices in movies_data
    sorted_scores = [pagerank_scores.get(i, 0) for i in range(len(movies_data))]
    
    # Normalize scores to [0, 1]
    total_score = sum(pagerank_scores.values())
    normalized_scores = np.array([score / total_score for score in sorted_scores])

    return normalized_scores


def get_movie_scores(movie_titles, path='./Datasets/Movies_Merged.csv', method='tfidf', top_n=10):
    # Load movie data
    movies_data = pd.read_csv(path)
    
    # Apply feature combination
    movies_data["text_features"] = movies_data.apply(combine_features_with_weights, axis=1)
    scaled_numerical_features = preprocess_numerical_features(movies_data) 
    
    if method == "tfidf":
        # Tokenize for TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['text_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        graph = build_graph_tfidf_numeric(cosine_sim, scaled_numerical_features)
        print(graph)
        
    elif method == "bm25":
        # Tokenize for BM25
        tokenized_features = [doc.split() for doc in movies_data["text_features"]]
        bm25 = BM25Okapi(tokenized_features)
        graph = build_graph_bm25_numeric(movies_data, bm25, scaled_numerical_features)
        print(graph)
      
    indices = pd.Series(movies_data.index, index=movies_data["Title"]).drop_duplicates()
    start_nodes = [indices.get(movie) for movie in movie_titles if indices.get(movie) is not None]

    if not start_nodes:
        return f"None of the movies in {movie_titles} were found in the dataset."

    recommendations = personalized_pagerank_multiple(movies_data, graph, start_nodes=start_nodes)
    return recommendations

# Example usage
'''
example_movie1 = ['The Godfather', 'The Traitor']
scores2 = get_movie_scores(example_movie1, method = "tfidf")
movies_data = pd.read_csv('./Datasets/Movies_Merged.csv')
movie_titles = movies_data["Title"]
top_indices_content_based = scores2.argsort()[::-1][:10]
print("Top 10 Movie Recommendations based on Content Based: {} = {}".format(movie_titles[top_indices_content_based], scores2[top_indices_content_based]))
'''