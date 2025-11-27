import json
import pandas as pd
import re
from tqdm import tqdm
from rapidfuzz import process, fuzz
import time
import os
import random

# Input Paths
REDIAL_RAW_TRAIN = '/content/drive/MyDrive/CS224W_Project/data/raw/redial/train_data.jsonl'
REDIAL_RAW_TEST = '/content/drive/MyDrive/CS224W_Project/data/raw/redial/test_data.jsonl'
REDIAL_MOVIES_PATH = '/content/drive/MyDrive/CS224W_Project/data/raw/redial/movies_with_mentions.csv'

# MovieLens Paths
MOVIELENS_MOVIES_PATH = '/content/drive/MyDrive/CS224W_Project/data/raw/movielens/movies.csv'
MOVIELENS_LINKS_PATH = '/content/drive/MyDrive/CS224W_Project/data/raw/movielens/links.csv'

# Output Paths (We now create 3 files)
OUTPUT_TRAIN = '/content/drive/MyDrive/CS224W_Project/data/processed/redial/redial_train_converted.tsv'
OUTPUT_DEV   = '/content/drive/MyDrive/CS224W_Project/data/processed/redial/redial_dev_converted.tsv'
OUTPUT_TEST  = '/content/drive/MyDrive/CS224W_Project/data/processed/redial/redial_test_converted.tsv'

# Parameters
FUZZY_THRESHOLD = 90
DEV_SPLIT_RATIO = 0.10 # 10% of training data goes to Dev
RANDOM_SEED = 42

def normalize_title(title):
    if pd.isna(title): return ""
    title = str(title).lower()
    if title.endswith(', the'):
        title = 'the ' + title[:-5]
    title = re.sub(r'[^\w\s]', '', title)
    return title.strip()

def create_title_to_imdb_map():
    print("Building Title -> IMDb Map...")
    ml_movies = pd.read_csv(MOVIELENS_MOVIES_PATH)
    ml_links = pd.read_csv(MOVIELENS_LINKS_PATH, dtype={'imdbId': str})
    merged = pd.merge(ml_movies, ml_links, on='movieId')

    title_map = {}
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        norm_title = normalize_title(row['title'])
        imdb_id = 'tt' + str(row['imdbId']).zfill(7)
        title_map[norm_title] = imdb_id
    return title_map

def process_dialogues(dialogues, output_path, title_to_imdb, redial_id_to_name, valid_titles, split_name):
    print(f"\n--- Processing {split_name} ({len(dialogues)} conversations) ---")

    processed_rows = []
    matches_exact = 0
    matches_fuzzy = 0
    matches_missed = 0
    fuzzy_cache = {}

    for dialogue in tqdm(dialogues):
        conv_id = f"redial_{dialogue['conversationId']}"

        # --- A. Flatten Text ---
        full_text = ""
        movie_mentions = dialogue['movieMentions']
        if isinstance(movie_mentions, list): movie_mentions = {} # Fix bad data

        for msg in dialogue['messages']:
            text = msg['text']
            for m_id, m_name in movie_mentions.items():
                if m_name:
                    text = text.replace(f"@{m_id}", m_name)
            full_text += text + " "
        full_text = full_text.strip()

        # --- B. Extract Edges ---
        seeker_labels = dialogue['initiatorQuestions']
        recommender_labels = dialogue['respondentQuestions']

        if isinstance(seeker_labels, list): seeker_labels = {}
        if isinstance(recommender_labels, list): recommender_labels = {}

        for movie_id_str, labels in seeker_labels.items():
            movie_id = int(movie_id_str)

            rec_info = recommender_labels.get(movie_id_str)
            if not isinstance(rec_info, dict): rec_info = {}
            is_suggestion = rec_info.get('suggested', 0) == 1
            if not is_suggestion: continue

            liked = labels.get('liked')
            if liked == 1: outcome = 'accept_rating_good'
            elif liked == 0: outcome = 'reject'
            else: continue

            # --- C. Map ID ---
            movie_name = redial_id_to_name.get(movie_id)
            if not movie_name: continue

            norm_name = normalize_title(movie_name)

            # 1. Exact Match
            imdb_id = title_to_imdb.get(norm_name)

            if imdb_id:
                matches_exact += 1
            else:
                # 2. Fuzzy Match
                if norm_name in fuzzy_cache:
                    imdb_id = fuzzy_cache[norm_name]
                    if imdb_id: matches_fuzzy += 1
                    else: matches_missed += 1
                else:
                    result = process.extractOne(norm_name, valid_titles, scorer=fuzz.token_sort_ratio)
                    if result:
                        best_match, score = result[0], result[1]
                        if score >= FUZZY_THRESHOLD:
                            imdb_id = title_to_imdb[best_match]
                            fuzzy_cache[norm_name] = imdb_id
                            matches_fuzzy += 1
                        else:
                            fuzzy_cache[norm_name] = None
                            matches_missed += 1
                    else:
                         fuzzy_cache[norm_name] = None
                         matches_missed += 1

            if imdb_id:
                processed_rows.append({
                    'conversation_id': conv_id,
                    'text': full_text,
                    'movie_id': imdb_id,
                    'outcome': outcome,
                })

    # Save Result
    df_final = pd.DataFrame(processed_rows)
    print(f"Results for {split_name}:")
    print(f"  Exact Matches: {matches_exact}")
    print(f"  Fuzzy Matches: {matches_fuzzy}")
    print(f"  Missed: {matches_missed}")
    print(f"  Total Output Rows: {len(df_final)}")

    df_final.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to {output_path}")

def process_all():
    # 1. Load Maps
    title_to_imdb = create_title_to_imdb_map()
    valid_titles = list(title_to_imdb.keys())

    print("Loading ReDial movie map...")
    redial_movies = pd.read_csv(REDIAL_MOVIES_PATH)
    redial_id_to_name = dict(zip(redial_movies['movieId'], redial_movies['movieName']))

    # 2. Load and Split Training Data
    print(f"Loading {REDIAL_RAW_TRAIN}...")
    with open(REDIAL_RAW_TRAIN, 'r') as f:
        all_train_lines = [json.loads(line) for line in f]

    # Shuffle and Split
    random.seed(RANDOM_SEED)
    random.shuffle(all_train_lines)

    split_idx = int(len(all_train_lines) * (1 - DEV_SPLIT_RATIO))
    train_dialogues = all_train_lines[:split_idx]
    dev_dialogues = all_train_lines[split_idx:]

    print(f"Split complete: {len(train_dialogues)} Train, {len(dev_dialogues)} Dev")

    # 3. Process Train Split
    process_dialogues(train_dialogues, OUTPUT_TRAIN, title_to_imdb, redial_id_to_name, valid_titles, "TRAIN")

    # 4. Process Dev Split
    process_dialogues(dev_dialogues, OUTPUT_DEV, title_to_imdb, redial_id_to_name, valid_titles, "DEV")

    # 5. Process Test Data
    if os.path.exists(REDIAL_RAW_TEST):
        with open(REDIAL_RAW_TEST, 'r') as f:
            test_dialogues = [json.loads(line) for line in f]
        process_dialogues(test_dialogues, OUTPUT_TEST, title_to_imdb, redial_id_to_name, valid_titles, "TEST")
    else:
        print(f"Warning: {REDIAL_RAW_TEST} not found.")

if __name__ == "__main__":
    process_all()
