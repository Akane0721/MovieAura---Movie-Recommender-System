# MovieAura
This project aims to develop a personalized movie recommender system that delivers tailored recommendations based on user viewing history, genre preferences, and similar user preferences. This recommender can enhance user satisfaction by simplifying the content discovery process and minimizing search time.

## How to Run

1. Start the Django server:
   ```bash
   python manage.py runserver
   ```
   Then, open http://127.0.0.1:8000/ in your browser to access the home page.

2. Enter movie titles one by one:
- Type a movie title in the input field and click `Add` after each entry.
- Once all desired titles are added, click `Submit` to generate the recommendation report.

![](display/homepage.png)

3. View and navigate:

- Review the 5 most recommended movies in a pie chart.
- Click `Return to Home` to go back to the home page.

![](display/resultpage.png)
## Team Members
@akane0721  
@NikolaRaicevic2001  
@rishabhk153  
@Wendi Tan  
@Yi-Hsuan Wen

## Datasets
[Full TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)  
[16000+ Movies (Metacritic)](https://www.kaggle.com/datasets/kashifsahil/16000-movies-1910-2024-metacritic)
## File Structure
- **`main.py`**: Top module which combines submodule.
- **`Data_Processing.py`**: Merging different datasets and extract features.
- **`Data_Visualization.ipynb`**: Jupyter Notebook for data visualization
### Submodules
1. **`Submodule_01_Image_Based.py`**
2. **`Submodule_02_Content_Based.py`**
3. **`Submodule_03_Graph_Based.py`**
4. **`Submodule_04_Transformer.py`**

![](display/mod.png)
   
## Third-Party Modules
The project utilizes the following Python libraries:
- **Data Visualization**:
  - `seaborn`: For creating advanced, statistical plots.
  - `matplotlib`: For generating standard visualizations and customizing plots.
- **Machine Learning and Deep Learning**:
  - `tensorflow` or `torch`: For transformer-based architectures.
  - `scikit-learn`: For TF-IDF vectorization, scaling, and cosine similarity.
  - `sklearn.metrics`: Used to get the precision_score, recall_score, f1_score during the training.
- **Image Processing**:
  - `torchvision` and `keras.applications`: For ResNet-50 embeddings.
- **Graph-Based Algorithms**:
  - `networkx`: For graph construction and PageRank implementation.

## Steps to Run
1. Download `movie_features_merge_new.npy` and `Movie_Embeddings.pkl` from "https://drive.google.com/drive/folders/1pQiwsuW7wkabhcWuuibn4Weroiicl75S?usp=sharing"
2. `python main.py`

For custom inputs : update `movie_sequence` in `main.py`

# Main.py
Gathers all the movie scores from four submodules and weights them according to the relevance of each submodule to derive a final score to be sorted. From the sorted list the first five movies are chosen as the user recommendations.

# Data_Processing.py
Contains a class that processes dataset returning different fields that could be relevant for submodules. The Movies_Merged.csv file is created that merges all the relevant fields from the 16000-movies-1910-2024-metacritic and tmdb-movies-dataset-2023-930k-movies datasets. The two datasets were merged according to the movie title match from both datasets.

# Submodule_01_Image_Based.py 
This submodule implements a Hybrid Content-Based Filtering System to recommend movies by integrating visual, genre, and textual attributes. It leverages multiple feature types and calculates similarity scores to identify movies similar to user input.  

### Key Features
1. Feature Combination
   Combines multiple attributes of a movie (e.g., visual embeddings, genres, descriptions) into a unified feature matrix for robust similarity calculations.
   - Visual Embeddings (ResNet-50): Captures visual features from movie posters.
   - Genres: Encoded as binary features.
   - Descriptions: Processed using TF-IDF vectorization.
This ensures recommendations consider both thematic and visual similarity.

2. Data Normalization
   Applies MinMaxScaler to scale visual, genre, and textual features to a uniform range. This ensures equal importance across different feature types.

3. Similarity Calculation
   - Computes cosine similarity scores based on the combined feature matrix:
   - Measures the similarity between movies using visual, textual, and genre features.
   - Supports aggregated similarity scores for multiple input movie titles.
4. Flexible Input
   Accepts one or more movie titles as input:
   - Matches input movies against the dataset.
   - Returns a probabilities of all movies in the dataset to be recommended. 

### How It Works
1. Feature Extraction: Extract visual embeddings, encode genres, and vectorize descriptions using TF-IDF.
2. Data Normalization: Normalize extracted features for uniformity.
3. Feature Combination: Combine visual, genre, and text features into a single feature matrix.
4. Similarity Computation: Calculate cosine similarity scores between the input movies and all movies in the dataset.
5. Recommendation Generation: Return aggregated similarity scores and a ranked list of recommendations for user-specified movies.

# Submodule_02_Content_Based.py
This submodule implements a **Content-Based Filtering** system to recommend movies based on weighted features. It enhances movie attributes and calculates similarity scores to identify similar movies.

### Key Features
1. **Feature Combination with Weights**  
   Combines multiple attributes of a movie (e.g., genres, keywords, description) into a weighted string for better similarity calculations. Default weights are:
   - **Genre**: 2  
   - **Keywords**: 3  
   - **Summary, Director, and others**: 1  

2. **TF-IDF Vectorization**  
   Processes combined features using TF-IDF vectorization to represent movies as feature vectors.

3. **Similarity Calculation**  
   Computes cosine similarity scores between movies based on their feature vectors. Supports aggregated similarity scores for multiple input movie titles.
   
4. **Flexible Input**  
   Accepts one or more movie titles and computes similarity scores against all movies in the dataset.

### How It Works
1. Combine weighted features for each movie using attributes like genres, keywords, description, tagline, director, etc.
2. Vectorize the combined features using TF-IDF.
3. Compute cosine similarity scores between input movies and the dataset.
4. Return aggregated similarity scores for user-specified movie titles.

# **Submodule_03_Graph_Based.py**
This submodule implements a **Graph-Based Filtering system** combining **BM25/TF-IDF similarity** with the **PageRank algorithm** to recommend movies, integrating both **textual** and **numeric features**.

### Key Features
1. Text Features: Uses textual attributes with weighted importance:  
- **Genres** (2), **Keywords** (3)  
- **Description, Tagline, Written by, Directed by, Language, Companies** (1 each)

2. Numeric Features: Uses normalized numeric attributes:  
- **Release Year** (0.1), **Runtime** (0.1)  

### How it works
1. Text Similarity:
- **TF-IDF Vectorization**: Uses TF-IDF to process combined movie features, similar to Submodule_02 and Submodule_01. 
- **BM25 Scoring**: Leverages BM25, a robust ranking function that emphasizes term relevance by accounting for term frequency and document length.

2. Numeric Similarity:
- **Normalization**: Scales numeric features using `StandardScaler`.  
- **Cosine Similarity**: Computes cosine similarity between normalized numeric features to quantify similarity.

3. Graph Construction & PageRank:
- **Graph Nodes**: Treats movies as **nodes** and assigns **edge weights** based on combined BM25/TF-IDF and numeric similarity scores. 
- **PageRank**: Computes **global importance scores** for each movie. Iteratively refines rankings based on graph connectivity, emphasizing both direct similarity and network-wide relationships. 

# Submodule_04_Transformer.py
This submodule implements **Transformer Encoder** system to recommend movies based on numerical features. The model used is SASRec/BERT4Rec based Transformer-based architecture designed specifically for recommendation tasks. The model uses self-attention mechanism to model user sequence and can handle both sequential dependencies (past movies) and incorporate information or context based embeddings.

 - **Input Format**: The list of movies with their specific numerical embeddings including Adult movie, Genres, Production Country, Keywords, Description, Rating, Popularity, and Release Year. The Adult category was binary either true/false; Genres and Production Country was one-hot encoded; Rating, Popularity, and Release Year was normalized so that the features are in the range of 0 to 1; Keywords and Description features were extracted using pre-trained language model BERT to create a dense vector representation.

 - **Output Format**: The model generates sequence of logits which are scores for each movie to be corrolated with the input sequence of the movies. The softmax function is later used to generate a probability distribution over the potential next movies based on the sequence and contextual information.

