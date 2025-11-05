import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import torch.nn.functional as F
from scipy.sparse import csr_matrix

class MovieLensProcessor:
    def __init__(self, data_dir='ml-32m/'):
        self.data_dir = data_dir

    def create_tag_embeddings(self, file = 'tags.csv'):
        tags = pd.read_csv(f'{self.data_dir}/{file}')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        tags = tags.dropna(subset=['tag'])
        movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ';'.join(x)).reset_index()
        movie_tags.columns = ['movieId', 'all_tags']

        
        embeddings = model.encode(
            movie_tags['all_tags'].tolist(),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            show_progress_bar=True
        )
        movie_tags['tag_embedding'] = list(embeddings)

        movie_tags.to_parquet(os.path.join('processed_data', 'movie_tags_embeddings.parquet'))

    def initialize_user_embeddings(self, high_ratings, movie_embeddings, user_to_idx, movie_id_to_idx):
        """
        Initialize users as weighted average of movies they rated highly
        """
        n_users = len(user_to_idx)
        n_movies = movie_embeddings.shape[0]
        embedding_dim = movie_embeddings.shape[1]
        
        # Create sparse user-movie rating matrix
        user_indices = high_ratings['user_idx'].values
        movie_indices = high_ratings['movie_idx'].values
        ratings = high_ratings['rating'].values
        
        # Normalize ratings per user (for weighted average)
        user_rating_sums = high_ratings.groupby('user_idx')['rating'].sum()
        normalized_ratings = ratings / user_rating_sums.loc[user_indices].values
        
        # Create sparse matrix: users x movies (with normalized ratings as values)
        rating_matrix = csr_matrix(
            (normalized_ratings, (user_indices, movie_indices)),
            shape=(n_users, n_movies)
        )
        
        # Matrix multiplication: (n_users x n_movies) @ (n_movies x embedding_dim)
        # This computes weighted average in one operation!
        user_embeddings = rating_matrix @ movie_embeddings
        
        # Convert to torch
        user_embeddings = torch.tensor(user_embeddings, dtype=torch.float)
        
        # Normalize to match scale of movie embeddings
        movie_emb_tensor = torch.tensor(movie_embeddings, dtype=torch.float)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1) * movie_emb_tensor.norm(dim=1).mean()
        
        print(f"User embeddings shape: {user_embeddings.shape}")
        return user_embeddings
    
    def process(self):
        movies = pd.read_csv(f'{self.data_dir}/movies.csv')

        ## add tag embeddings to movies df       
        movie_tags = pd.read_parquet(os.path.join('processed_data', 'movie_tags_embeddings.parquet')) 
        movies = movies.merge(
            movie_tags[['movieId', 'tag_embedding']], 
            on='movieId', 
            how='inner'
        )

        ratings = pd.read_csv(f'{self.data_dir}/ratings.csv')

        movies['movie_idx'] = range(len(movies))
        movie_id_to_idx = dict(zip(movies['movieId'], movies['movie_idx']))
            
        genre_set = set()
        for genres_str in movies['genres']:
            if isinstance(genres_str, str) and genres_str != '(no genres listed)':
                genre_set.update(genres_str.split('|'))
        genre_set = sorted(genre_set)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        genre_embeddings = model.encode(
            genre_set,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            show_progress_bar=True
        )       

        genre_to_idx = {g: idx for idx, g in enumerate(genre_set)}
        
        movie_genre_edges = []
        for _, row in movies.iterrows():
            if isinstance(row['genres'], str):
                for genre in row['genres'].split('|'):
                    if genre in genre_to_idx:
                        movie_genre_edges.append([row['movie_idx'], genre_to_idx[genre]])
        
        movie_genre_edges = torch.tensor(movie_genre_edges, dtype=torch.long).t()
        genre_movie_edges = torch.stack([movie_genre_edges[1], movie_genre_edges[0]], dim=0)
        
        #users
        high_ratings = ratings[ratings['rating'] >= 4.0].copy()
         # Only keep ratings for movies we have
        high_ratings = high_ratings[high_ratings['movieId'].isin(movie_id_to_idx.keys())]

        unique_users = high_ratings['userId'].unique()
        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

        high_ratings['user_idx'] = high_ratings['userId'].map(user_to_idx)
        high_ratings['movie_idx'] = high_ratings['movieId'].map(movie_id_to_idx)

        high_ratings = high_ratings.dropna(subset=['user_idx', 'movie_idx'])
        
        user_movie_edges = torch.tensor([
            high_ratings['user_idx'].values,
            high_ratings['movie_idx'].values
        ], dtype=torch.long)
        movie_user_edges = torch.stack([user_movie_edges[1], user_movie_edges[0]], dim=0)


        data = HeteroData()
        
        # Movie nodes with tag embeddings
        movie_embeddings = np.stack(movies['tag_embedding'].values)
        
        data['movie'].num_nodes = len(movies)
        data['movie'].movie_id = torch.tensor(movies['movieId'].values, dtype=torch.long)
        data['movie'].movie_name = movies['title'].tolist()
        data['movie'].x = torch.tensor(movie_embeddings, dtype=torch.float)

        # User nodes
        data['user'].num_nodes = len(user_to_idx)
        data['user'].user_id = torch.tensor(list(user_to_idx.keys()), dtype=torch.long)
        data['user'].x = self.initialize_user_embeddings(
            high_ratings, 
            movie_embeddings, 
            user_to_idx, 
            movie_id_to_idx
        )
        
        # Genre nodes
        data['genre'].num_nodes = len(genre_to_idx)
        data['genre'].genre_name = list(genre_to_idx.keys())
        data['genre'].x = torch.tensor(genre_embeddings, dtype=torch.float)
        
        # Add edges
        data['movie', 'has_genre', 'genre'].edge_index = movie_genre_edges
        data['genre', 'has_movie', 'movie'].edge_index = genre_movie_edges
        
        data['user', 'rated_high', 'movie'].edge_index = user_movie_edges
        data['movie', 'rated_by', 'user'].edge_index = movie_user_edges
        # edge attributes (ratings)
        data['user', 'rated_high', 'movie'].edge_attr = torch.tensor(
            high_ratings['rating'].values, 
            dtype=torch.float
        ).unsqueeze(1)
        data['movie', 'rated_by', 'user'].edge_attr = torch.tensor(
            high_ratings['rating'].values, 
            dtype=torch.float   
        ).unsqueeze(1)

        # Print summary
        print("\n=== Graph Summary ===")
        print(f"Movies (with tags): {data['movie'].num_nodes}")
        print(f"Users: {data['user'].num_nodes}")
        print(f"Genres: {data['genre'].num_nodes}")
        print(f"Movie features shape: {data['movie'].x.shape}")
        print(f"Movie-Genre edges: {data['movie', 'has_genre', 'genre'].edge_index.shape[1]}")
        print(f"Genre-Movie edges: {data['genre', 'has_movie', 'movie'].edge_index.shape[1]}")
        print(f"User-Movie edges: {data['user', 'rated_high', 'movie'].edge_index.shape[1]}")
        print(f"Movie-User edges: {data['movie', 'rated_by', 'user'].edge_index.shape[1]}")
        
        # Save the processed data
        print("\nSaving processed data...")
        torch.save(data, os.path.join('processed_data', 'hetero_graph.pt'))
        
        # Save mappings for later use
        mappings = {
            'movie_id_to_idx': movie_id_to_idx,
            'user_to_idx': user_to_idx,
            'genre_to_idx': genre_to_idx
        }
        torch.save(mappings, os.path.join('processed_data', 'id_mappings.pt'))

        high_ratings.to_parquet(os.path.join('processed_data', 'high_ratings.parquet'))
        movies.to_parquet(os.path.join('processed_data', 'movies_with_tags.parquet'))



# Run it
processor = MovieLensProcessor('data/ml-32m/')
#processor.create_tag_embeddings()
processor.process()