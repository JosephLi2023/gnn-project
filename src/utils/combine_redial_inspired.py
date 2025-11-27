import torch
import pandas as pd
import json
import pickle
import re
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import os
from tqdm import tqdm

class ConversationGraphPipeline:
    """
    A class to build the new 'conversation' nodes and 'outcome' edge features.
    UPDATED: Handles merged INSPIRED + ReDial datasets.
    """
    def __init__(self, train_paths, dev_paths, test_paths, inspired_movie_map_path,
                 movielens_links_path, mappings_path):

        # UPDATED: Expect lists of paths for each split
        # This allows you to pass [inspired_train.tsv, redial_train.tsv]
        self.paths = {
            "train": train_paths if isinstance(train_paths, list) else [train_paths],
            "dev": dev_paths if isinstance(dev_paths, list) else [dev_paths],
            "test": test_paths if isinstance(test_paths, list) else [test_paths]
        }
        self.inspired_movie_map_path = inspired_movie_map_path
        self.movielens_links_path = movielens_links_path
        self.mappings_path = mappings_path

        # UPDATED: We focus on the binary signals you requested
        # Maps string labels to integer class indices
        self.outcome_to_int_map = {
            'accept_rating_good': 1, # Liked (1)
            'reject': 0              # Did not like (0)
        }
        self.num_edge_features = len(self.outcome_to_int_map)


    def _clean_text(self, text):
        """A helper function to clean up raw text."""
        if not isinstance(text, str):
            return None
        text = re.sub(r"[^a-zA-Z0-9\s.,?'!-]", '', text)
        return text if text else None


    def generate_and_build_data(self, mode, model_name='all-MiniLM-L6-v2', output_pt_path_prefix=None):
        """
        Loads, merges, and processes data for a specific split.
        """
        print(f"--- Running Pipeline for {mode.upper()} SPLIT ---")

        # 1. Load ID Mappings
        print("Loading ID maps...")
        df_movie_map = pd.read_csv(self.inspired_movie_map_path, sep='\t', on_bad_lines='skip')
        links_df = pd.read_csv(self.movielens_links_path, dtype={'imdbId': str})
        mappings = torch.load(self.mappings_path, weights_only=False)
        movie_id_to_idx = mappings['movie_id_to_idx']

        # Map: INSPIRED_Video_ID -> IMDb_ID
        inspired_id_to_imdb = pd.Series(
            df_movie_map['imdb_id'].values,
            index=df_movie_map['video_id']
        ).to_dict()

        # Map: IMDb_ID -> MovieLens_ID
        imdb_to_movielens_id = {}
        for _, row in links_df.iterrows():
            imdb_id = 'tt' + row['imdbId'].zfill(7)
            imdb_to_movielens_id[imdb_id] = row['movieId']

        # 2. Load and Merge Data Files
        print(f"Loading and merging files for {mode}...")
        df_list = []
        for path in self.paths[mode]:
            if os.path.exists(path):
                print(f"  - Reading {path}")
                df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
                df_list.append(df)
            else:
                print(f"  - Warning: File not found {path}")

        if not df_list:
            print("No data found. Exiting.")
            return None, None, None

        df_split = pd.concat(df_list, ignore_index=True)
        print(f"Total rows: {len(df_split)}")

        # 3. Pre-processing
        print(f"Pre-scanning conversations...")

        # Handle column name differences (INSPIRED vs ReDial/Merged)
        # outcome_col: 'fine_label' in INSPIRED, 'outcome' in processed ReDial
        outcome_col = 'outcome' if 'outcome' in df_split.columns else 'fine_label'
        # id_col: 'dialog_id' in INSPIRED, 'conversation_id' in processed ReDial
        id_col = 'conversation_id' if 'conversation_id' in df_split.columns else 'dialog_id'

        if outcome_col not in df_split.columns or id_col not in df_split.columns:
             print(f"Error: Required columns ({id_col}, {outcome_col}) not found in merged dataframe.")
             print(f"Available columns: {df_split.columns}")
             return None, None, None

        df_split['clean_text'] = df_split['text'].apply(self._clean_text)
        df_split = df_split.dropna(subset=['clean_text', 'movie_id', outcome_col])

        # Aggregate text by conversation (combine all utterances)
        agg_text_map = df_split.groupby(id_col)['clean_text'].apply(' '.join).to_dict()

        # IMPORTANT: We iterate rows because a conversation can have MULTIPLE recommendations
        # (The previous .groupby().last() logic would miss multi-recommendation convos)
        good_conv_text_map = {} # Store unique conversations text to encode
        valid_edges = []        # Store list of (conv_id, movie_node_idx, label)

        # Use tqdm to show progress
        for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
            conv_id = row[id_col]
            raw_movie_id = row['movie_id']
            outcome_label = row[outcome_col]

            # Filter by outcome first (we only want 'accept_rating_good' or 'reject')
            if outcome_label not in self.outcome_to_int_map:
                continue

            # --- UPDATED ID MAPPING LOGIC ---
            imdb_id = None

            # Case A: It's an INSPIRED Video ID (look it up)
            if raw_movie_id in inspired_id_to_imdb:
                imdb_id = inspired_id_to_imdb[raw_movie_id]
            # Case B: It's already an IMDb ID (from ReDial, e.g. 'tt1234567')
            elif str(raw_movie_id).startswith('tt'):
                imdb_id = raw_movie_id

            if not imdb_id: continue

            # Map IMDb -> MovieLens -> Graph Node Index
            movielens_id = imdb_to_movielens_id.get(imdb_id)
            if movielens_id:
                movie_node_idx = movie_id_to_idx.get(movielens_id)

                if movie_node_idx is not None:
                    # Save text (only need to do this once per convo, but overwriting is fine/cheap)
                    good_conv_text_map[conv_id] = agg_text_map[conv_id]
                    # Save Edge
                    valid_edges.append([conv_id, movie_node_idx, outcome_label])

        print(f"Found {len(good_conv_text_map)} unique valid conversations.")
        print(f"Found {len(valid_edges)} valid training edges.")

        if not good_conv_text_map:
            return None, None, None

        # 4. Encode Text
        print(f"Loading sentence-transformer model: {model_name}...")
        model = SentenceTransformer(model_name)

        conv_ids = list(good_conv_text_map.keys())
        texts = list(good_conv_text_map.values())

        print(f"Encoding {len(texts)} conversations...")
        embeddings = model.encode(texts, show_progress_bar=True)

        conversation_x = torch.tensor(embeddings, dtype=torch.float)
        # Map conversation ID string -> new tensor index (0, 1, 2...)
        conv_id_to_idx = {conv_id: i for i, conv_id in enumerate(conv_ids)}

        # 5. Build Tensors
        edge_list_for_index = []
        attr_list_for_one_hot = []

        for conv_id, movie_node_idx, outcome_label in valid_edges:
            # Get the new index for this conversation
            conv_idx = conv_id_to_idx[conv_id]

            # Add edge (source, target)
            edge_list_for_index.append([conv_idx, movie_node_idx])

            # Add attribute label
            attr_list_for_one_hot.append(self.outcome_to_int_map[outcome_label])

        conv_movie_edge_index = torch.tensor(edge_list_for_index, dtype=torch.long).t().contiguous()
        attr_labels = torch.tensor(attr_list_for_one_hot, dtype=torch.long)

        # One-hot encode the edge attributes
        conv_movie_edge_attr = F.one_hot(attr_labels, num_classes=self.num_edge_features).float()

        print(f"\nFinal shapes:")
        print(f"  conversation_x: {conversation_x.shape}")
        print(f"  edge_index: {conv_movie_edge_index.shape}")
        print(f"  edge_attr: {conv_movie_edge_attr.shape}")

        # 6. Save
        if output_pt_path_prefix:
            conv_x_path = f"{output_pt_path_prefix}_{mode}_conversation_x.pt"
            edge_index_path = f"{output_pt_path_prefix}_{mode}_conv_movie_edge_index.pt"
            edge_attr_path = f"{output_pt_path_prefix}_{mode}_conv_movie_edge_attr.pt"

            print(f"Saving to {output_pt_path_prefix}_{mode}_*...")
            torch.save(conversation_x, conv_x_path)
            torch.save(conv_movie_edge_index, edge_index_path)
            torch.save(conv_movie_edge_attr, edge_attr_path)

        return conversation_x, conv_movie_edge_index, conv_movie_edge_attr
