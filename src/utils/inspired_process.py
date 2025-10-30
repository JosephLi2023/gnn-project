import torch
import pandas as pd
import json
import pickle
import re
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class ConversationGraphPipeline:
    """
    A class to build the new 'conversation' nodes
    and 'outcome' edge features.
    It pre-filters to ensure NO isolated nodes are created.
    """
    def __init__(self, raw_inspired_path, inspired_movie_map_path, 
                 movielens_links_path, mappings_path):
        
        self.raw_inspired_path = raw_inspired_path
        self.inspired_movie_map_path = inspired_movie_map_path
        self.movielens_links_path = movielens_links_path
        self.mappings_path = mappings_path
        
        self.conversation_x = None
        self.conv_movie_edge_index = None
        self.conv_movie_edge_attr = None
        self.conv_id_to_idx = {}
        
        self.outcome_to_int_map = {
            'accept_rating_good': 0,
            'accept_rating_mod': 1,
            'accept_uninterested': 2,
            'accept_others': 3,
            'reject': 4
        }
        self.num_edge_features = len(self.outcome_to_int_map)

    
    def _clean_text(self, text):
        """A helper function to clean up raw text."""
        if not isinstance(text, str):
            return None
        text = re.sub(r"[^a-zA-Z0-9\s.,?'!-]", '', text)
        return text if text else None

    
    def generate_and_build_data(self, model_name='all-MiniLM-L6-v2', output_pt_path_prefix=None):
        """
        Loads all data, finds valid mappable conversations,
        aggregates text, creates embeddings, and builds the
        conversation_x, edge_index, and edge_attr.
        """        
        print("Loading all data and map files...")
        df_train = pd.read_csv(self.raw_inspired_path, sep='\t', on_bad_lines='skip')
        df_movie_map = pd.read_csv(self.inspired_movie_map_path, sep='\t', on_bad_lines='skip')
        links_df = pd.read_csv(self.movielens_links_path, dtype={'imdbId': str})
        mappings = torch.load(self.mappings_path, weights_only=False)
        movie_id_to_idx = mappings['movie_id_to_idx'] 

        inspired_id_to_imdb = pd.Series(
            df_movie_map['imdb_id'].values, 
            index=df_movie_map['video_id']
        ).to_dict()
        
        imdb_to_movielens_id = {}
        for _, row in links_df.iterrows():
            imdb_id = 'tt' + row['imdbId'].zfill(7) 
            imdb_to_movielens_id[imdb_id] = row['movieId']
            
        print("Pre-scanning conversations to find valid, mappable edges...")
        
        good_conv_text_map = {}
        valid_edges = []
        
        df_train['clean_text'] = df_train['text'].apply(self._clean_text)
        df_train = df_train.dropna(subset=['clean_text', 'movie_id', 'fine_label'])
        
        agg_text_map = df_train.groupby('dialog_id')['clean_text'].apply(' '.join).to_dict()
        # Get the last movie and outcome for each conversation
        conv_movie_map = df_train.groupby('dialog_id')['movie_id'].last().to_dict()
        conv_outcome_map = df_train.groupby('dialog_id')['fine_label'].last().to_dict()
        
        for conv_id, inspired_movie_id in conv_movie_map.items():
            outcome_label = conv_outcome_map.get(conv_id)
            
            if (inspired_movie_id == '#NAME?' or 
                inspired_movie_id not in inspired_id_to_imdb or
                outcome_label not in self.outcome_to_int_map):
                continue
            
            imdb_id = inspired_id_to_imdb[inspired_movie_id]
            movielens_id = imdb_to_movielens_id.get(imdb_id)
            
            if movielens_id:
                movie_node_idx = movie_id_to_idx.get(movielens_id)
                
                if movie_node_idx is not None:
                    good_conv_text_map[conv_id] = agg_text_map[conv_id]
                    valid_edges.append([conv_id, movie_node_idx, outcome_label])

        print(f"Found {len(good_conv_text_map)} valid conversations that map to a movie.")

        if not good_conv_text_map:
            print("No valid conversations found. Exiting.")
            return

        print(f"Loading sentence-transformer model: {model_name}...")
        model = SentenceTransformer(model_name)
        
        conv_ids = list(good_conv_text_map.keys())
        texts = list(good_conv_text_map.values())

        print(f"Encoding {len(texts)} conversations...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        self.conversation_x = torch.tensor(embeddings, dtype=torch.float)
        self.conv_id_to_idx = {conv_id: i for i, conv_id in enumerate(conv_ids)}

        edge_list_for_index = []
        attr_list_for_one_hot = []

        for conv_id, movie_node_idx, outcome_label in valid_edges:
            conv_idx = self.conv_id_to_idx[conv_id]
            
            edge_list_for_index.append([conv_idx, movie_node_idx])
            
            # Add the integer label to the attr list
            attr_list_for_one_hot.append(self.outcome_to_int_map[outcome_label])

        # Create the edge_index tensor
        self.conv_movie_edge_index = torch.tensor(edge_list_for_index, dtype=torch.long).t().contiguous()
        
        # Create a tensor of the integer labels
        attr_labels = torch.tensor(attr_list_for_one_hot, dtype=torch.long)
        # Convert to one-hot
        self.conv_movie_edge_attr = F.one_hot(attr_labels, num_classes=self.num_edge_features).float()


        print(f"Final conversation feature matrix shape: {self.conversation_x.shape}")
        print(f"Final new edge_index shape: {self.conv_movie_edge_index.shape}")
        print(f"Final new edge_attr shape: {self.conv_movie_edge_attr.shape}")

        if output_pt_path_prefix:
            conv_x_path = f"{output_pt_path_prefix}_conversation_x.pt"
            edge_index_path = f"{output_pt_path_prefix}_conv_movie_edge_index.pt"
            edge_attr_path = f"{output_pt_path_prefix}_conv_movie_edge_attr.pt" 
            
            print(f"Saving conversation features to {conv_x_path}...")
            torch.save(self.conversation_x, conv_x_path)
            
            print(f"Saving new edge index to {edge_index_path}...")
            torch.save(self.conv_movie_edge_index, edge_index_path)
            
            print(f"Saving new edge attributes to {edge_attr_path}...")
            torch.save(self.conv_movie_edge_attr, edge_attr_path)

        return self.conversation_x, self.conv_movie_edge_index, self.conv_movie_edge_attr
