import os
import glob
import string
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # Progress bar
from collections import Counter
from functools import lru_cache

# Load SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Feature Extraction Functions
@lru_cache(maxsize=None)
def extract_syntactic_features(cell: str):
    """
    Extracts syntactic features from a single table cell.
    Caches results for repeated cells to save recomputation.
    """
    s = str(cell)
    length = len(s)
    if length == 0:
        # Return a list of zeros with the same number of features (39 in this case)
        return [0] * 39

    # Compute once and reuse
    words = s.split()
    num_words = len(words)
    num_digits = sum(c.isdigit() for c in s)
    num_uppercase = sum(c.isupper() for c in s)
    num_lowercase = sum(c.islower() for c in s)
    num_punctuation = sum(c in string.punctuation for c in s)
    num_whitespace = sum(c.isspace() for c in s)
    num_special_chars = sum((not c.isalnum()) and (not c.isspace()) for c in s)

    proportion_uppercase = num_uppercase / length
    proportion_digits = num_digits / length
    proportion_punctuation = num_punctuation / length
    proportion_whitespace = num_whitespace / length

    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    longest_word_length = max((len(word) for word in words), default=0)
    shortest_word_length = min((len(word) for word in words), default=0)
    proportion_words = num_words / length

    contains_email = "@" in s
    contains_url = any(substr in s for substr in ["http://", "https://", "www."])
    contains_hashtag = "#" in s
    currency_symbols = {'$', 'â¬', 'Â£', 'Â¥'}
    contains_currency = any(sym in s for sym in currency_symbols)

    is_numeric = s.isdigit()
    is_alpha = s.isalpha()
    is_alphanumeric = s.isalnum()
    is_capitalized = s.istitle()

    # Use Counter for frequency counts
    counts = Counter(s)
    shannon_entropy = -sum((cnt / length) * np.log2(cnt / length) for cnt in counts.values())
    unique_chars = len(counts)
    proportion_vowels = sum(c in "aeiouAEIOU" for c in s) / length
    is_palindrome = s == s[::-1]
    repeating_chars = sum(s[i] == s[i - 1] for i in range(1, length))
    repeating_words = sum(words[i] == words[i - 1] for i in range(1, len(words)))
    first_char_type = ord(s[0])
    last_char_type = ord(s[-1])
    most_frequent_char = max(counts.values(), default=0)
    least_frequent_char = min(counts.values(), default=0)
    digit_frequency = num_digits / length
    punctuation_frequency = num_punctuation / length
    whitespace_frequency = num_whitespace / length
    char_diversity_ratio = unique_chars / length
    num_alpha_sequences = sum(1 for word in words if word.isalpha())

    return [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace, num_special_chars,
        proportion_uppercase, proportion_digits, proportion_punctuation, proportion_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length, proportion_words,
        contains_email, contains_url, contains_hashtag, contains_currency, is_numeric, is_alpha,
        is_alphanumeric, is_capitalized, shannon_entropy, unique_chars, proportion_vowels, is_palindrome,
        repeating_chars, repeating_words, first_char_type, last_char_type, most_frequent_char,
        least_frequent_char, digit_frequency, punctuation_frequency, whitespace_frequency,
        char_diversity_ratio, num_alpha_sequences
    ]


def combine_features_for_table(table: pd.DataFrame):
    """
    Processes a table by computing semantic embeddings (batched) and syntactic features for each cell,
    then combines them into a single feature vector per cell.
    The final output is a PyTorch tensor in channel-first format -> (channels, num_rows, num_cols),
    where channels = 39 syntactic features + 384 semantic embedding dimensions = 423
    """
    # Convert table cells to strings and flatten
    cell_values = table.astype(str).to_numpy().flatten().tolist()

    # Batch process semantic embeddings
    semantic_embeddings = model.encode(cell_values, batch_size=32, convert_to_numpy=True)

    combined_features = []
    for i, cell in enumerate(cell_values):
        syntactic_features = extract_syntactic_features(cell)
        combined = np.hstack([syntactic_features, semantic_embeddings[i]])
        combined_features.append(combined)

    num_rows, num_cols = table.shape
    # Reshape back to the original table structure -> (num_rows, num_cols, 423)
    table_features = np.array(combined_features).reshape(num_rows, num_cols, -1)

    # Convert the numpy array to a pytorch tensor
    table_tensor = torch.tensor(table_features, dtype=torch.float32)

    # Permute dimensions to channel-first format -> (channels, num_rows, num_cols)
    table_tensor = table_tensor.permute(2, 0, 1)
    return table_tensor

folder_path = "non_relational_tables"
file_pattern = os.path.join(folder_path, "*.csv")
file_paths = glob.glob(file_pattern)

results = []
for file_path in tqdm(file_paths, desc="Processing tables"):
    file_name = os.path.basename(file_path)
    if file_name.endswith("unstack.csv"):
        label = "unstack"
    elif file_name.endswith("transpose.csv"):
        label = "transpose"
    else:
        print(f"Skipping file '{file_name}' (unknown transformation).")
        continue

    try:
        table = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading '{file_name}': {e}")
        continue

    # Process the table into a tensor with shape (423, num_rows, num_cols)
    table_tensor = combine_features_for_table(table)
    results.append({
        "file": file_name,
        "label": label,
        "features": table_tensor
    })

print(f"Processing complete. Extracted features for {len(results)} tables.")

# Stack all the table tensors into one tensor with shape: (N, 423, num_rows, num_cols) where N is the number of tables
feature_tensors = [res["features"] for res in results]
all_tensor_tables = torch.stack(feature_tensors)

label_map = {"unstack": 0, "transpose": 1}
all_labels = torch.tensor([label_map[res["label"]] for res in results], dtype=torch.long)

#print("Overall dataset tensor shape:", all_tensor_tables.shape)
#print("Overall labels tensor shape:", all_labels.shape)

# Create a PyTorch dataset and DataLoader.
dataset = TensorDataset(all_tensor_tables, all_labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# print("\n--- Sample Batch from DataLoader ---")
# for batch_inputs, batch_labels in dataloader:
#     print("Batch inputs shape:", batch_inputs.shape)  # Expected: (batch_size, 423, num_rows, num_cols)
#     print("Batch labels shape:", batch_labels.shape)  # Expected: (batch_size,)
#     break
