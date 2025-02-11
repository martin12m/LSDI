import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import string
from sentence_transformers import SentenceTransformer

# ---- Operators and Inverse Operators ----
import pandas as pd

def unstack_dataframe(df):
    """
    Implements the Unstack Operator (Inverse of Stack).
    - Identifies the last categorical column as the pivot.
    - Uses pivoting to transform rows into columns.
    - Ensures numeric values remain consistent across transformations.
    """
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

        if len(categorical_cols) < 1 or len(numeric_cols) == 0:
            raise ValueError("Table must contain categorical and numeric columns for unstacking.")

        pivot_column = categorical_cols[-1]  
        index_cols = categorical_cols[:-1]  

        # 🔥 Ensure Unique Index Before Pivoting
        df['_unique_id'] = df.groupby(index_cols).cumcount()

        # Pivot (Unstack Logic)
        unstacked_df = df.pivot(index=index_cols + ['_unique_id'], columns=pivot_column, values=numeric_cols[0])
        unstacked_df = unstacked_df.reset_index().drop(columns=['_unique_id'])

        print("✅ Unstack Transformation Successful!")
        return unstacked_df

    except Exception as e:
        print(f"❌ Error during unstacking: {e}")
        return df

    
def transpose_dataframe_inverse(df):
    """
    Transposes a DataFrame while maintaining semantic meaning of rows and columns.
    """

    try:
        print("\n🔄 Applying Transpose Operator...")

        # Ensure we have column names
        df.columns = [str(col) for col in df.columns]

        # Set a meaningful index before transposing
        transposed_df = df.set_index(df.columns[0]).T

        # Reset index and ensure column naming consistency
        transposed_df = transposed_df.reset_index()
        transposed_df.columns = ['Index'] + list(transposed_df.columns[1:])

        print(f"✅ Transposed DataFrame Shape: {transposed_df.shape}")
        return transposed_df

    except Exception as e:
        print(f"⚠ Error during transpose: {e}")
        return df

def apply_random_inverse_operator(df):
    operators = {
        "unstack": unstack_dataframe,
        "transpose": transpose_dataframe_inverse
    }
    operator_name = random.choice(list(operators.keys()))
    transformed_df = operators[operator_name](df)
    return transformed_df, operator_name

# ---- Feature Extraction ----

def extract_syntactic_features(cell):
    """
    Extracts 39 syntactic features dynamically from cell values.
    - Works for both string and numeric inputs.
    - Ensures correctness by handling special cases.
    - Aligns with research paper's approach.
    """
    if not isinstance(cell, str):
        cell = str(cell)  # Convert non-string inputs to string safely

    length = len(cell)
    
    # Handle edge case where length is 0
    if length == 0:
        return [0] * 39

    num_digits = sum(c.isdigit() for c in cell)
    num_uppercase = sum(c.isupper() for c in cell)
    num_lowercase = sum(c.islower() for c in cell)
    num_punctuation = sum(c in string.punctuation for c in cell)
    num_whitespace = sum(c.isspace() for c in cell)
    num_special_chars = sum(not c.isalnum() and not c.isspace() for c in cell)

    # Proportions (Ensure safe division)
    proportion_uppercase = round(num_uppercase / length, 3) if length > 0 else 0
    proportion_digits = round(num_digits / length, 3) if length > 0 else 0
    proportion_punctuation = round(num_punctuation / length, 3) if length > 0 else 0
    proportion_whitespace = round(num_whitespace / length, 3) if length > 0 else 0

    # Word-level features
    words = cell.split()
    num_words = len(words)
    avg_word_length = round(sum(len(word) for word in words) / num_words, 3) if num_words > 0 else 0
    longest_word_length = max((len(word) for word in words), default=0)
    shortest_word_length = min((len(word) for word in words), default=0)
    proportion_words = round(num_words / length, 3) if length > 0 else 0

    # Special patterns
    contains_email = "@" in cell
    contains_url = any(substr in cell for substr in ["http://", "https://", "www."])
    contains_hashtag = "#" in cell
    contains_at_symbol = "@" in cell  # Redundant but kept for research alignment
    is_numeric = cell.isdigit()
    is_alpha = cell.isalpha()
    is_alphanumeric = cell.isalnum()
    is_capitalized = cell.istitle()

    # Shannon entropy (Fixed for empty or single-character strings)
    try:
        shannon_entropy = -sum(
            (cell.count(c) / length) * np.log2(cell.count(c) / length)
            for c in set(cell)
        )
        shannon_entropy = round(shannon_entropy, 3) if length > 1 else 0
    except ValueError:
        shannon_entropy = 0  # Handles log(0) issue safely

    # Unique character ratio
    unique_chars = len(set(cell))
    proportion_vowels = round(sum(c in "aeiouAEIOU" for c in cell) / length, 3) if length > 0 else 0

    # Pattern-based features
    is_palindrome = cell == cell[::-1]
    repeating_chars = sum(cell[i] == cell[i - 1] for i in range(1, length))
    
    # Fix for repeating words in single-word inputs
    repeating_words = sum(words[i] == words[i - 1] for i in range(1, len(words))) if num_words > 1 else 0

    # Character types (ASCII values of first and last char)
    first_char_type = ord(cell[0]) if length > 0 else 0
    last_char_type = ord(cell[-1]) if length > 0 else 0

    # Most/Least frequent character occurrences
    most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
    least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)

    # Frequency-based features
    digit_frequency = round(num_digits / length, 3) if length > 0 else 0
    punctuation_frequency = round(num_punctuation / length, 3) if length > 0 else 0
    whitespace_frequency = round(num_whitespace / length, 3) if length > 0 else 0
    char_diversity_ratio = round(unique_chars / length, 3) if length > 0 else 0
    num_alpha_sequences = sum(1 for part in words if part.isalpha())

    return [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, 
        num_whitespace, num_special_chars, proportion_uppercase, 
        proportion_digits, proportion_punctuation, proportion_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length, 
        proportion_words, contains_email, contains_url, contains_hashtag, 
        contains_at_symbol, is_numeric, is_alpha, is_alphanumeric, is_capitalized, 
        shannon_entropy, unique_chars, proportion_vowels, is_palindrome, 
        repeating_chars, repeating_words, first_char_type, last_char_type, 
        most_frequent_char, least_frequent_char, digit_frequency, 
        punctuation_frequency, whitespace_frequency, char_diversity_ratio, 
        num_alpha_sequences
    ]
# def extract_syntactic_features(cell):
#     if not isinstance(cell, str):
#         cell = str(cell)
#     length = len(cell)
#     num_digits = sum(c.isdigit() for c in cell)
#     num_uppercase = sum(c.isupper() for c in cell)
#     num_lowercase = sum(c.islower() for c in cell)
#     num_punctuation = sum(c in string.punctuation for c in cell)
#     num_whitespace = sum(c.isspace() for c in cell)
#     num_special_chars = sum(not c.isalnum() and not c.isspace() for c in cell)
#     proportion_uppercase = num_uppercase / length if length > 0 else 0
#     proportion_digits = num_digits / length if length > 0 else 0
#     proportion_punctuation = num_punctuation / length if length > 0 else 0
#     proportion_whitespace = num_whitespace / length if length > 0 else 0
#     num_words = len(cell.split())
#     avg_word_length = sum(len(word) for word in cell.split()) / num_words if num_words > 0 else 0
#     longest_word_length = max((len(word) for word in cell.split()), default=0)
#     shortest_word_length = min((len(word) for word in cell.split()), default=0)
#     proportion_words = num_words / length if length > 0 else 0
#     contains_email = "@" in cell
#     contains_url = any(substr in cell for substr in ["http://", "https://", "www."])
#     contains_hashtag = "#" in cell
#     contains_at_symbol = "@" in cell
#     is_numeric = cell.isdigit()
#     is_alpha = cell.isalpha()
#     is_alphanumeric = cell.isalnum()
#     is_capitalized = cell.istitle()
#     shannon_entropy = -sum(p * np.log2(p) for p in [cell.count(c) / length for c in set(cell)]) if length > 0 else 0
#     unique_chars = len(set(cell))
#     proportion_vowels = sum(c in "aeiouAEIOU" for c in cell) / length if length > 0 else 0
#     is_palindrome = cell == cell[::-1]
#     repeating_chars = sum(cell[i] == cell[i - 1] for i in range(1, length))
#     repeating_words = sum(cell.split()[i] == cell.split()[i - 1] for i in range(1, len(cell.split())))
#     first_char_type = ord(cell[0]) if length > 0 else 0
#     last_char_type = ord(cell[-1]) if length > 0 else 0
#     most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
#     least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)
#     digit_frequency = num_digits / length if length > 0 else 0
#     punctuation_frequency = num_punctuation / length if length > 0 else 0
#     whitespace_frequency = num_whitespace / length if length > 0 else 0
#     char_diversity_ratio = unique_chars / length if length > 0 else 0
#     num_alpha_sequences = sum(1 for part in cell.split() if part.isalpha())

#     return [
#         length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace, num_special_chars,
#         proportion_uppercase, proportion_digits, proportion_punctuation, proportion_whitespace,
#         num_words, avg_word_length, longest_word_length, shortest_word_length, proportion_words,
#         contains_email, contains_url, contains_hashtag, contains_at_symbol, is_numeric, is_alpha,
#         is_alphanumeric, is_capitalized, shannon_entropy, unique_chars, proportion_vowels, is_palindrome,
#         repeating_chars, repeating_words, first_char_type, last_char_type, most_frequent_char,
#         least_frequent_char, digit_frequency, punctuation_frequency, whitespace_frequency,
#         char_diversity_ratio, num_alpha_sequences
#     ]

# def combine_features_for_row(row, model):
#     combined_row_features = []
#     for cell in row:
#         syntactic_features = extract_syntactic_features(cell)
#         semantic_features = model.encode([str(cell)], convert_to_tensor=False)[0]
#         combined_features = np.hstack([syntactic_features, semantic_features])
#         combined_row_features.append(combined_features)
#     return np.array(combined_row_features)
def combine_features_for_row(row, model):
    """
    Extracts syntactic and semantic features while maintaining table structure.
    """
    combined_row_features = []
    
    for cell in row:
        # Extract syntactic features
        syntactic_features = extract_syntactic_features(cell)
        
        # Extract semantic features using Sentence-BERT
        if isinstance(cell, str):
            semantic_features = model.encode([cell], convert_to_tensor=False)[0]
        else:
            semantic_features = np.zeros(384)  # If cell is numeric, no embedding needed
        
        # Concatenate both features
        combined_features = np.hstack([syntactic_features, semantic_features])
        combined_row_features.append(combined_features)
    
    return np.array(combined_row_features)


def resize_dataframe(df, target_rows=101, target_cols=50):

    try:
        print("\n🛠 Debug: Resizing DataFrame...")

        # Calculate how many rows/columns to keep
        num_rows, num_cols = df.shape

        # If too many rows, slice it down
        if num_rows > target_rows:
            df = df.iloc[:target_rows, :]
        # If too few rows, repeat existing rows
        elif num_rows < target_rows:
            repeat_factor = (target_rows // num_rows) + 1
            df = pd.concat([df] * repeat_factor, ignore_index=True).iloc[:target_rows, :]
            

        # If too many columns, slice it down
        if num_cols > target_cols:
            df = df.iloc[:, :target_cols]
        # If too few columns, repeat existing columns
        elif num_cols < target_cols:
            repeat_factor = (target_cols // num_cols) + 1
            df = pd.concat([df] * repeat_factor, axis=1).iloc[:, :target_cols]
            

        print(f"✅ Resized DataFrame Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"⚠ Error during resizing: {e}")
        return df
    
def main():
    # Load relational Table
    relational_data = pd.read_csv("e-commerce_1.csv")
    
    
    
    # Apply a random inverse operator to the relational table
    non_relational_table, operator_name = apply_random_inverse_operator(relational_data)
    print(f"Applied random inverse operator: {operator_name}")
    
    non_relational_table.to_csv("non_relational_data.csv", index=False)
    print(non_relational_table.shape) 
    print(f"Shape of transformed table: {non_relational_table.shape}") 
    if non_relational_table.empty:
        print("Transformed table is empty. Exiting.")
        return

    resized_df = resize_dataframe(non_relational_table)
    resized_df.to_csv('resized_table.csv', index=False)
    print(f"Shape of resized table: {resized_df.shape}")
    
    # Load Sentence-BERT model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Extract features
    all_features = []
    for row in resized_df.itertuples(index=False):
        row_features = combine_features_for_row(row, model)
        if row_features.size > 0:
            all_features.append(row_features)

    if not all_features:
        print("No features extracted. Exiting.")
        return

    # Create feature tensor
    final_features = np.stack(all_features)
    print(f"Feature Tensor Shape: {final_features.shape}")
    print(final_features[0, 0, :10])

if __name__ == "__main__":
    main()
