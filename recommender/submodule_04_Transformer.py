import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score

import pandas as pd
import numpy as np

from submodule_04_Constants import (SEQ_LEN,EMBEDDING_DIM,BATCH_SIZE,NUM_LAYERS,NUM_HEADS,DROPOUT_RATE,EPOCHS,INPUT_DIM,NUM_MOVIES)

#############################################################
### Define Transformer-Based Model (SASRec/BERT4Rec-like) ###
#############################################################
class TransformerRecModel(nn.Module):
    def __init__(self, num_movies=NUM_MOVIES, input_dim=INPUT_DIM, sequence_len=SEQ_LEN, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, sequence_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout_rate, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, num_movies)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, features):
        # Project features into embedding space
        projected_features = self.input_projection(features)

        # Add positional embeddings
        embedded = projected_features + self.positional_embedding[:, :features.size(1), :]
        
        # Apply transformer encoder
        transformer_output = self.encoder(embedded)
        
        # Output logits for next movie (last position)
        logits = self.output_layer(transformer_output[:, -1])
        return logits

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        features, targets = batch
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(features) 
        loss = criterion(output, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)

            # Get model output
            output = model(features)  # Shape: (BATCH_SIZE, NUM_MOVIES)

            # Calculate loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Get predictions
            _, predicted_indices = torch.max(output, dim=1)  # Get the indices of the max log-probability
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_indices.cpu().numpy())

            # Count correct predictions
            correct_predictions += (predicted_indices == targets).sum().item()
            total_samples += targets.size(0)  # Number of samples in this batch

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

def get_movie_scores(movie_sequence, model_path='NN_Models_Transformer/TransformerRecModel_30.pth'):
    # Load the saved model
    model = TransformerRecModel(num_movies=NUM_MOVIES, input_dim=INPUT_DIM, sequence_len=SEQ_LEN, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load movie embeddings
    feature_vectors, _ = get_movie_embeddings('Dataset_Processed/Movie_Embeddings.pkl',movie_sequence)

    model.eval()
    with torch.no_grad():
        # Ensure we have a fixed length (SEQ_LEN)
        if len(feature_vectors) < SEQ_LEN:
            # Create padding with zeros
            padding = np.zeros((SEQ_LEN - len(feature_vectors), feature_vectors.shape[1]))
            feature_vectors = np.vstack([feature_vectors, padding])
        else:
            feature_vectors = feature_vectors[:SEQ_LEN]
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor(feature_vectors, dtype=torch.float).unsqueeze(0).to(device)  # Shape (1, SEQ_LEN, INPUT_DIM)
        
        # Get model output
        logits = model(input_tensor)
        
        # Apply softmax to get probabilities
        output = F.softmax(logits, dim=1)

        return output.squeeze().cpu().numpy() 
    
def get_movie_embeddings(path, movie_sequence):
    # Loading movie embeddings from pickle file
    movie_embeddings = pd.read_pickle(path)

    # Dictionary mapping movie titles to indices
    movie_to_index = {title: int(idx) for idx, title in zip(movie_embeddings['ID'].values, movie_embeddings['Title'].values)}
    input_indices = [movie_to_index[movie] for movie in movie_sequence] # Convert movie titles to indices

    # Extract features from pickle file
    features = np.array(list(movie_embeddings['Description_Embedding']))    # Description embeddings
    keywords = np.array(list(movie_embeddings['Keyword_Embedding']))        # Keyword embeddings
    genres = movie_embeddings.loc[:, "Action":"Western"].values             # Genre embeddings
    countries = movie_embeddings.loc[:, "Afghanistan":"Zimbabwe"].values    # Country embeddings
    other_features = movie_embeddings[["Adult","Normalized_Release_Year","Normalized_Rating","Normalized_Popularity"]].values
    feature_vectors = np.concatenate([features[input_indices], keywords[input_indices], genres[input_indices], countries[input_indices], other_features[input_indices]], axis=1)

    return feature_vectors, movie_embeddings

if __name__ == "__main__":
    # Example sequence of movie titles (representing a user's watch history)
    movie_sequence = ['Three Colors: Red', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction',
                       'The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring', 
                       'The Lord of the Rings: The Two Towers', 'The Matrix', 'The Dark Knight Rises', 'Inception', 'Interstellar',
                       'Django Unchained', 'The Prestige', 'The Departed', 'The Green Mile', 'The Lion King','The Truman Show',
                       'The Silence of the Lambs', 'The Usual Suspects', 'The Pianist', 'The Sixth Sense']

    print("Getting movie scores...")
    probabilities = get_movie_scores(movie_sequence)
    print(probabilities.shape)

    # Load movie embeddings
    _, movie_embeddings = get_movie_embeddings('Dataset_Processed/Movie_Embeddings.pkl',movie_sequence)

    # Get top 5 movie recommendations
    top_indices = probabilities.argsort()[::-1][:5]
    print("Top 5 Movie Recommendations:")
    print(movie_embeddings['Title'].values[top_indices])