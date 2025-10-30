import torch
import pandas as pd
import json
import re
import pickle
from sentence_transformers import SentenceTransformer

class TextFeaturePipeline:
    """
    A class to handle the full pipeline from raw text to an aligned
    feature matrix for the GNN.
    """
    def __init__(self, raw_inspired_path, inspired_movie_map_path,
                 movielens_links_path, mappings_path, graph_path):

        self.raw_inspired_path = raw_inspired_path         # train_set.tsv
        self.inspired_movie_map_path = inspired_movie_map_path # file with movie_id (INSPIRED) to imdb_id
        self.movielens_links_path = movielens_links_path   # links.csv
        self.mappings_path = mappings_path # id_mappings.pt
        self.graph_path = graph_path       # hetero_graph.pt

        self.imdb_to_text = {}
        self.imdb_to_embeddings = {}


    def generate_embeddings(self, model_name='all-MiniLM-L6-v2', output_json_path=None, output_pickle_path=None):
        """
        Step 1: Loads raw INSPIRED2 data, aggregates all text by imdb_id,
                and generates embeddings for each movie.
        """

        print("Loading raw INSPIRED2 files...")
        df_train = pd.read_csv(self.raw_inspired_path, sep='\t')
        df_movie_map = pd.read_csv(self.inspired_movie_map_path, sep='\t')


        inspired_id_to_imdb = pd.Series(
            df_movie_map['imdb_id'].values,
            index=df_movie_map['video_id']
        ).to_dict()

        print(inspired_id_to_imdb)

        print(f"Aggregating text from {len(df_train)} records...")

        for _, row in df_train.iterrows():
            movie_id = row['movie_id']

            if movie_id == '#NAME?' or movie_id not in inspired_id_to_imdb:
                continue

            imdb_id = inspired_id_to_imdb[movie_id]
            text = row['text']

            if not isinstance(text, str):
                continue
            
            text = re.sub(r"[^a-zA-Z0-9\s.,?'!-]", '', text)

            if not text:
                continue

            if imdb_id not in self.imdb_to_text:
                self.imdb_to_text[imdb_id] = text
            else:
                self.imdb_to_text[imdb_id] += " " + text

        print(f"Aggregated text for {len(self.imdb_to_text)} unique movies.")

        print(f"Loading sentence-transformer model: {model_name}...")
        model = SentenceTransformer(model_name)

        imdb_ids = list(self.imdb_to_text.keys())
        texts = list(self.imdb_to_text.values())

        print("Encoding all texts...")
        embeddings = model.encode(texts, show_progress_bar=True)

        self.imdb_to_embeddings = {imdb_id: emb.tolist() for imdb_id, emb in zip(imdb_ids, embeddings)}
        print("Embedding generation complete.")

        if output_json_path:
            print(f"Saving embeddings to {output_json_path}...")
            with open(output_json_path, 'w') as f:
              json.dump(self.imdb_to_embeddings, f)
        if output_pickle_path:
            with open(output_pickle_path, 'wb') as f:
              pickle.dump(self.imdb_to_embeddings, f)

        return self.imdb_to_embeddings


    def build_feature_matrix(self, output_pt_path=None):
        """
        Step 2: Builds the final 'movie_x' tensor.
        """

        if not self.imdb_to_embeddings:
            print("Error: Embeddings not found. Please run .generate_embeddings() first.")
            return None

        print("Loading graph files...")
        mappings = torch.load(self.mappings_path, weights_only=False)
        movie_id_to_idx = mappings['movie_id_to_idx']

        data = torch.load(self.graph_path, weights_only=False)
        num_movie_nodes = data['movie'].num_nodes
        embedding_dim = len(next(iter(self.imdb_to_embeddings.values())))

        print(f"Total movie nodes in graph: {num_movie_nodes}")
        print(f"Embedding dimension is: {embedding_dim}")

        print("Loading MovieLens links.csv...")
        links_df = pd.read_csv(self.movielens_links_path, dtype={'imdbId': str})
        movielens_id_to_imdb = {}
        for _, row in links_df.iterrows():
            imdb_id = 'tt' + row['imdbId'].zfill(7)
            movielens_id_to_imdb[row['movieId']] = imdb_id

        print(f"Building 'movie_x' matrix with shape ({num_movie_nodes}, {embedding_dim})...")
        movie_x = torch.zeros(num_movie_nodes, embedding_dim)

        movies_found = 0
        movies_not_found = 0

        for movielens_id, node_index in movie_id_to_idx.items():
            imdb_id = movielens_id_to_imdb.get(movielens_id)
            if imdb_id:
                embedding = self.imdb_to_embeddings.get(imdb_id)
                if embedding:
                    movie_x[node_index] = torch.tensor(embedding, dtype=torch.float)
                    movies_found += 1
                else:
                    movies_not_found += 1
            else:
                movies_not_found += 1

        print(f"Found and placed {movies_found} movie embeddings.")
        print(f"Could not find embeddings for {movies_not_found} movies (left as zeros).")

        if output_pt_path:
            print(f"Saving final matrix to {output_pt_path}...")
            torch.save(movie_x, output_pt_path)

        return movie_x
