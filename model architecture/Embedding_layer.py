# import pandas as pd
# import numpy as np
# from faker import Faker
# import random
# import string
# from sentence_transformers import SentenceTransformer

# #Generate Random Syntactic DataFrame
# def generate_syntactic_dataframe(num_rows=100, num_cols=50):
#     """Generates a DataFrame with diverse syntactic data."""
#     fake = Faker()
#     data = {}
#     for col in range(num_cols):
#         data[f"Column{col+1}"] = [
#             random.choice([
#                 str(random.randint(100, 9999)),       # Numbers
#                 fake.email(),                         # Emails
#                 fake.date_this_decade().isoformat(),  # Dates
#                 fake.word().upper(),                  # Uppercase
#                 fake.word().lower(),                  # Lowercase
#                 ''.join(random.choices(string.punctuation, k=5)), # Punctuation
#                 fake.word().capitalize() + str(random.randint(10, 99)) # Mixed
#             ]) for _ in range(num_rows)
#         ]
#     return pd.DataFrame(data)

# def extract_syntactic_features(cell):
#     """Extracts syntactic features for a single cell."""
#     cell = str(cell)
#     length = len(cell)
#     num_digits = sum(c.isdigit() for c in cell)
#     num_uppercase = sum(c.isupper() for c in cell)
#     num_lowercase = sum(c.islower() for c in cell)
#     num_punctuation = sum(c in string.punctuation for c in cell)
#     num_whitespace = sum(c.isspace() for c in cell)
#     words = cell.split()
#     num_words = len(words)
#     avg_word_length = np.mean([len(word) for word in words]) if words else 0
#     longest_word_length = max([len(word) for word in words], default=0)
#     shortest_word_length = min([len(word) for word in words], default=0)
#     num_special_chars = sum(1 for c in cell if not c.isalnum() and not c.isspace())
#     ratio_upper_to_lower = num_uppercase / (num_lowercase + 1e-5)
#     ratio_digits_to_length = num_digits / (length + 1e-5)
#     ratio_punctuation_to_length = num_punctuation / (length + 1e-5)
#     custom_features = [
#         len(set(cell)),                # Unique characters
#         cell.count('@'),               # Count of '@'
#         cell.count('#'),               # Count of '#'
#     ]

#     #Combine all features
#     syntactic_features = [
#         length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace,
#         num_words, avg_word_length, longest_word_length, shortest_word_length,
#         num_special_chars, ratio_upper_to_lower, ratio_digits_to_length, ratio_punctuation_to_length
#     ] + custom_features  # Add additional features here

#     #Ensure exactly 39 features (pad with zeros if needed)
#     return syntactic_features + [0] * (39 - len(syntactic_features))

# def compute_syntactic_features(df):
#     """Computes syntactic features for the entire table."""
#     features = []
#     for row in df.itertuples(index=False):
#         row_features = [extract_syntactic_features(cell) for cell in row]
#         features.extend(row_features)
#     return np.array(features)

# #Compute Semantic Features (384-Dimensional)
# def compute_semantic_features(df, model):
#     """Computes semantic features using Sentence-BERT."""
#     cells = df.values.flatten().astype(str)
#     embeddings = model.encode(cells, convert_to_tensor=False)  # Sentence-BERT embeddings
#     return embeddings

# #Combine Syntactic and Semantic Features
# def combine_features(syntactic_features, semantic_features):
#     """Combines syntactic and semantic features for each cell."""
#     return np.hstack([syntactic_features, semantic_features])  # Shape: (n*m, 423)

# def main():
#     #Define Table Dimensions
#     num_rows = 101  # Total rows (including header)
#     num_cols = 50   # Number of columns

#     #Generate Random Table Data
#     df = generate_syntactic_dataframe(num_rows=num_rows - 1, num_cols=num_cols)  # Exclude header row in generation
#     print("Sample DataFrame:")
#     print(df.head())  # Display first 5 rows


#     #Debugging within `compute_syntactic_features`
#     syntactic_features = compute_syntactic_features(df)
#     print("Syntactic Features Shape:", syntactic_features.shape)
#     print("First Cell Syntactic Features (Length):", len(syntactic_features[0]))
#     print("Syntactic Features:")
#     #Load Pre-trained Sentence-BERT Model
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#     #Compute Features
#     syntactic_features = compute_syntactic_features(df)  # Shape: (data_rows * num_cols, 39)
#     print("Syntactic Features Shape:", syntactic_features.shape)

#     semantic_features = compute_semantic_features(df, model)  # Shape: (data_rows * num_cols, 384)
#     print("Semantic Features Shape:", semantic_features.shape)

#     #Combine Syntactic and Semantic Features
#     combined_features = combine_features(syntactic_features, semantic_features)  # Shape: (data_rows * num_cols, 423)
#     print("Combined Features Shape:", combined_features.shape)

#     #Reshape to Tensor
#     data_rows = num_rows - 1  # Adjust for actual data rows
#     expected_size = data_rows * num_cols  # Total cells
#     if combined_features.shape[0] != expected_size:
#         raise ValueError(f"Feature size mismatch: Expected {expected_size} rows, got {combined_features.shape[0]}.")

#     feature_tensor = combined_features.reshape(data_rows, num_cols, 423)  # Adjusted for data_rows
#     print("\nFinal Feature Tensor:")
#     print(feature_tensor)
#     print("Shape:", feature_tensor.shape)

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
from faker import Faker
import random
import string
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate Random Syntactic DataFrame
def generate_syntactic_dataframe(num_rows=100, num_cols=50):
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

# 2. Extract Syntactic Features (39 Features)
def extract_syntactic_features(cell):
    """Extracts syntactic features for a single cell."""
    cell = str(cell)
    
    # Basic features
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
    ratio_whitespace_to_length = num_whitespace / (length + 1e-5)
    ratio_words_to_length = num_words / (length + 1e-5)
    
    # Custom features
    unique_chars = len(set(cell))
    num_vowels = sum(1 for c in cell if c in 'aeiouAEIOU')
    num_consonants = sum(1 for c in cell if c.isalpha() and c not in 'aeiouAEIOUaeiou')
    num_digits_before_non_digit = len(cell) - len(cell.lstrip('0123456789'))
    num_punct_before_alnum = len(cell) - len(cell.lstrip(string.punctuation + string.ascii_letters + string.digits))
    num_letters_before_digit = len(cell) - len(cell.lstrip(string.ascii_letters))
    
    num_uppercase_words = sum(1 for word in words if word.isupper())
    num_lowercase_words = sum(1 for word in words if word.islower())
    num_capitalized_words = sum(1 for word in words if word.istitle())
    
    contains_email = 1 if '@' in cell and '.' in cell else 0
    contains_url = 1 if 'http' in cell or 'www' in cell else 0
    contains_hashtag = 1 if '#' in cell else 0
    contains_at_symbol = 1 if '@' in cell else 0
    
    num_punctuation_after_first_word = sum(1 for c in cell.split(maxsplit=1)[-1] if c in string.punctuation)
    num_digits_after_first_word = sum(1 for c in cell.split(maxsplit=1)[-1] if c.isdigit())
    num_special_after_first_word = sum(1 for c in cell.split(maxsplit=1)[-1] if not c.isalnum() and not c.isspace())
    
    num_words_with_digits = sum(1 for word in words if any(c.isdigit() for c in word))
    num_words_with_special_chars = sum(1 for word in words if any(not c.isalnum() for c in word))
    
    contains_numerical_range = 1 if '-' in cell and any(c.isdigit() for c in cell) else 0
    contains_monetary_value = 1 if any(c in cell for c in ['$', '€', '£', '¥']) else 0
    num_capital_letters_in_word = sum(1 for c in max(words, key=len) if c.isupper())
    contains_non_alphanumeric = 1 if any(not c.isalnum() for c in cell) else 0
    num_digits_in_longest_word = sum(1 for c in max(words, key=len) if c.isdigit())
    
    # Combine all features
    syntactic_features = [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length,
        num_special_chars, ratio_upper_to_lower, ratio_digits_to_length, ratio_punctuation_to_length,
        ratio_whitespace_to_length, ratio_words_to_length, unique_chars, num_vowels, num_consonants,
        num_digits_before_non_digit, num_punct_before_alnum, num_letters_before_digit, num_uppercase_words,
        num_lowercase_words, num_capitalized_words, contains_email, contains_url, contains_hashtag,
        contains_at_symbol, num_punctuation_after_first_word, num_digits_after_first_word,
        num_special_after_first_word, num_words_with_digits, num_words_with_special_chars,
        contains_numerical_range, contains_monetary_value, num_capital_letters_in_word, contains_non_alphanumeric,
        num_digits_in_longest_word
    ]
    
    return syntactic_features

#  Extract Semantic Features (using Sentence-BERT)
def compute_semantic_features(df, model):
    """Compute the semantic features (embeddings) for the entire DataFrame."""
    cells = df.values.flatten().astype(str)
    embeddings = model.encode(cells, convert_to_tensor=False)  # Sentence-BERT embeddings
    return embeddings

# Combine Features (Syntactic + Semantic)
def combine_features(syntactic_features, semantic_features):
    """Combine the syntactic and semantic features into a single matrix."""
    return np.hstack([syntactic_features, semantic_features])

# Normalize the Features (Ensure consistent shape for further processing)
def normalize_table_features(features, target_cells=101, feature_dim=423):
    """Pads or truncates the features to match the desired shape of target_cells x feature_dim."""
    num_cells = features.shape[0]
    if num_cells > target_cells:
        features = features[:target_cells]
    elif num_cells < target_cells:
        padding = np.zeros((target_cells - num_cells, feature_dim))
        features = np.vstack([features, padding])
    return features


def main():
    #  Generate Random Table Data
    df = generate_syntactic_dataframe(num_rows=100, num_cols=50)
    print("Sample DataFrame:")
    print(df)
    
    #  Load Pre-trained Sentence-BERT Model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    #  Extract Features
    syntactic_features = np.array([extract_syntactic_features(cell) for row in df.itertuples(index=False) for cell in row])
    print("syntactic_features: ",syntactic_features.shape)
    
    #  Compute Semantic Features
    semantic_embeddings = compute_semantic_features(df, model)
    print("semantic_embeddings: ",semantic_embeddings.shape)
    
    #  Combine Features (Syntactic + Semantic)
    features = combine_features(syntactic_features, semantic_embeddings)
    print("combine_features :",features.shape)
    
    # #Normalize/Pad to 101 Cells
    # final_features = normalize_table_features(features, target_cells=101, feature_dim=423)
    
    # print("\nFinal Input Feature Matrix (101x423):")
    # print(final_features)
    # print("Shape:", final_features.shape)
    
    # final_features_reshaped = final_features.reshape(-1, final_features.shape[-1])
    
    # # Create DataFrame to save as CSV
    # final_features_df = pd.DataFrame(final_features_reshaped)
    # final_features_df.to_csv('final_features.csv', index=False, header=False)
    # print("\nFeatures saved to 'final_features.csv'.")

if __name__ == "__main__":
    main()

