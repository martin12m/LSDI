import pandas as pd
import numpy as np
from faker import Faker
import random
import string
from sentence_transformers import SentenceTransformer

# Generate Random Syntactic DataFrame
def generate_syntactic_dataframe(num_rows=10, num_cols=10):
    """Generates a DataFrame with diverse syntactic data."""
    fake = Faker()
    data = {}
    for col in range(num_cols):
        data[f"Column{col+1}"] = [
            random.choice([
                str(random.randint(100, 9999)),       # Numbers
                fake.email(),                         # Emails
                fake.date_this_decade().isoformat(),  # Dates
                fake.word().upper(),                  # Uppercase
                fake.word().lower(),                  # Lowercase
                ''.join(random.choices(string.punctuation, k=5)), # Punctuation
                fake.word().capitalize() + str(random.randint(10, 99)) # Mixed
            ]) for _ in range(num_rows)
        ]
    return pd.DataFrame(data)

#  Extract Syntactic Features (39 Features)
def extract_syntactic_features(cell):
    """Extracts syntactic features for a single cell."""
    cell = str(cell)
    length = len(cell)
    num_digits = sum(c.isdigit() for c in cell)
    num_uppercase = sum(c.isupper() for c in cell)
    num_lowercase = sum(c.islower() for c in cell)
    num_punctuation = sum(c in string.punctuation for c in cell)
    num_whitespace = sum(c.isspace() for c in cell)
    words = cell.split()
    num_words = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    longest_word_length = max([len(word) for word in words], default=0)
    shortest_word_length = min([len(word) for word in words], default=0)
    num_special_chars = sum(1 for c in cell if not c.isalnum() and not c.isspace())
    ratio_upper_to_lower = num_uppercase / (num_lowercase + 1e-5)
    ratio_digits_to_length = num_digits / (length + 1e-5)
    ratio_punctuation_to_length = num_punctuation / (length + 1e-5)
    return [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length,
        num_special_chars, ratio_upper_to_lower, ratio_digits_to_length, ratio_punctuation_to_length
    ] + [0] * (39 - 14)  # Padding to reach 39 features.

def compute_syntactic_features(df):
    """Computes syntactic features for the entire table."""
    features = []
    for row in df.itertuples(index=False):
        row_features = [extract_syntactic_features(cell) for cell in row]
        features.extend(row_features)
    return np.array(features)

# Compute Semantic Features (384-Dimensional)
def compute_semantic_features(df, model):
    """Computes semantic features using Sentence-BERT."""
    cells = df.values.flatten().astype(str)
    embeddings = model.encode(cells, convert_to_tensor=False)  # Sentence-BERT embeddings
    return embeddings

# Combine Syntactic and Semantic Features
def combine_features(syntactic_features, semantic_features):
    """Combines syntactic and semantic features for each cell."""
    return np.hstack([syntactic_features, semantic_features])  # Shape: (n*m, 423)


def main():
    # Define Table Dimensions
    num_rows = 101 
    num_cols = 50  

    # Generate Random Table Data
    df = generate_syntactic_dataframe(num_rows=num_rows, num_cols=num_cols)
    print("Sample DataFrame:")
    print(df)

    # Load Pre-trained Sentence-BERT Model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

   # Compute Features
    syntactic_features = compute_syntactic_features(df)  # Shape: (n * m, 39)
    semantic_features = compute_semantic_features(df, model)  # Shape: (n * m, 384)

    # Combine Syntactic and Semantic Features
    combined_features = combine_features(syntactic_features, semantic_features)  # Shape: (n * m, 423)

    # Reshape to Tensor 
    feature_tensor = combined_features.reshape(num_rows, num_cols, 423)

    print("\nFinal Feature Tensor:")
    print(feature_tensor)
    print("Shape:", feature_tensor.shape)  

if __name__ == "__main__":
    main()
