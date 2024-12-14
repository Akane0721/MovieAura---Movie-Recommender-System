import submodule_01_Image_Based
import submodule_02_Content_Based
import submodule_03_Graph_Based
import submodule_04_Transformer 

# Define weights for each module based on importance
weights = {
    'submodule_01': 50,
    'submodule_02': 10,
    'submodule_03': 1,
    'submodule_04': 100,
}

def main():
    # # Random Movies
    # movie_sequence = ['Three Colors: Red', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction',
    #                    'The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring', 
    #                    'The Lord of the Rings: The Two Towers', 'The Matrix', 'The Dark Knight Rises', 'Inception', 'Interstellar',
    #                    'Django Unchained', 'The Prestige', 'The Departed', 'The Green Mile', 'The Lion King','The Truman Show',
    #                    'The Silence of the Lambs', 'The Usual Suspects']
    
    # Action/Adventure/Fantasy Movies
    movie_sequence = ['The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring','The Lord of the Rings: The Two Towers',
                      'The Matrix', 'The Dark Knight Rises', 'Interstellar','Django Unchained', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight',
                      'The Hobbit: An Unexpected Journey', 'The Hobbit: The Desolation of Smaug', 'The Hobbit: The Battle of the Five Armies']

    # Load movie embeddings
    _, movie_embeddings = submodule_04_Transformer.get_movie_embeddings('Dataset_Processed/Movie_Embeddings.pkl',movie_sequence)
    movie_titles = movie_embeddings['Title'].values

    # Retrieve scores from each module
    scores1 = submodule_01_Image_Based.get_movie_scores(movie_sequence, feature_path='./PosterFeatures/movie_features.npy')
    scores2 = submodule_02_Content_Based.get_movie_scores(movie_sequence, path='./Datasets/Movies_Merged.csv')
    scores3 = submodule_03_Graph_Based.get_movie_scores(movie_sequence)
    scores4 = submodule_04_Transformer.get_movie_scores(movie_sequence, model_path='NN_Models_Transformer/TransformerRecModel_50.pth')
    
    print("Shapes of the scores:")
    print("Scores shape for Image Based:{}".format(scores1.shape))
    print("Scores shape for Content Based:{}".format(scores2.shape))
    print("Scores shape for Graph Based:{}".format(scores3.shape))
    print("Scores shape for Transformer:{}".format(scores4.shape))

    # Top 10 movies based on each submodule
    print("\nTop 10 Movie Recommendations based on each submodule:")
    top_indices_image_based = scores1.argsort()[::-1][:10]
    print("Top 10 Movie Recommendations based on Image Based: {} = {}".format(movie_titles[top_indices_image_based], scores1[top_indices_image_based]))
    top_indices_content_based = scores2.argsort()[::-1][:10]
    print("Top 10 Movie Recommendations based on Content Based: {} = {}".format(movie_titles[top_indices_content_based], scores2[top_indices_content_based]))
    top_indices_graph_based = scores3.argsort()[::-1][:10]
    print("Top 10 Movie Recommendations based on Graph Based: {} = {}".format(movie_titles[top_indices_graph_based], scores3[top_indices_graph_based]))
    top_indices_transformer = scores4.argsort()[::-1][:10]
    print("Top 10 Movie Recommendations based on Transformer: {} = {}".format(movie_titles[top_indices_transformer], scores4[top_indices_transformer]))

    # Compute weighted score for each movie
    weighted_scores = [
        scores1[i] * weights['submodule_01'] +
        scores2[i] * weights['submodule_02'] +
        scores3[i] * weights['submodule_03'] +
        scores4[i] * weights['submodule_04']
        for i in range(len(scores4))
    ]
    
    # Pair movie titles with their scores and filter out movies in the input sequence
    movie_recommendations = list(zip(movie_titles, weighted_scores))
    movie_recommendations = [movie for movie in movie_recommendations if movie[0] not in movie_sequence]

    # Sort movies by score in descending order
    top_movies = sorted(movie_recommendations, key=lambda x: x[1], reverse=True)[:5]
    
    # Display the top five recommended movies
    print("Top 5 Movie Recommendations:")
    for title, score in top_movies:
        print(f"{title}: {score:.4e}")

if __name__ == "__main__":
    main()

