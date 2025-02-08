import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import string
from sentence_transformers import SentenceTransformer

# ---- Operators and Inverse Operators ----
def unstack_dataframe(df):
    try:
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

        if len(non_numeric_cols) < 2 or len(numeric_cols) == 0:
            raise ValueError("Table must contain categorical and numeric columns for unstacking.")
        
        index_cols = non_numeric_cols[:-1]
        unstack_col = non_numeric_cols[-1]

        unstacked_df = df.pivot(index=index_cols, columns=unstack_col, values=numeric_cols[0])
        unstacked_df = unstacked_df.reset_index()

        return unstacked_df

    except Exception as e:
        print(f"Error during unstacking: {e}")
        return df

def transpose_dataframe_inverse(df):
    try:
        print("\n🛠 Debug: Transposing DataFrame...")

        # Ensure first column is categorical (Avoid numeric index issues)
        if df.dtypes[0] != 'object':
            df.columns = [f"Column_{i}" for i in range(df.shape[1])]

        # Transpose and reset index
        transposed_df = df.set_index(df.columns[0]).T.reset_index()

        # Print debug info
        print(f"✅ Transposed DataFrame Shape: {transposed_df.shape}")
        return transposed_df

    except Exception as error:
        print(f"⚠ Error during transpose inverse: {error}")
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
    if not isinstance(cell, str):
        cell = str(cell)
    length = len(cell)
    num_digits = sum(c.isdigit() for c in cell)
    num_uppercase = sum(c.isupper() for c in cell)
    num_lowercase = sum(c.islower() for c in cell)
    num_punctuation = sum(c in string.punctuation for c in cell)
    num_whitespace = sum(c.isspace() for c in cell)
    num_special_chars = sum(not c.isalnum() and not c.isspace() for c in cell)
    proportion_uppercase = num_uppercase / length if length > 0 else 0
    proportion_digits = num_digits / length if length > 0 else 0
    proportion_punctuation = num_punctuation / length if length > 0 else 0
    proportion_whitespace = num_whitespace / length if length > 0 else 0
    num_words = len(cell.split())
    avg_word_length = sum(len(word) for word in cell.split()) / num_words if num_words > 0 else 0
    longest_word_length = max((len(word) for word in cell.split()), default=0)
    shortest_word_length = min((len(word) for word in cell.split()), default=0)
    proportion_words = num_words / length if length > 0 else 0
    contains_email = "@" in cell
    contains_url = any(substr in cell for substr in ["http://", "https://", "www."])
    contains_hashtag = "#" in cell
    contains_at_symbol = "@" in cell
    is_numeric = cell.isdigit()
    is_alpha = cell.isalpha()
    is_alphanumeric = cell.isalnum()
    is_capitalized = cell.istitle()
    shannon_entropy = -sum(p * np.log2(p) for p in [cell.count(c) / length for c in set(cell)]) if length > 0 else 0
    unique_chars = len(set(cell))
    proportion_vowels = sum(c in "aeiouAEIOU" for c in cell) / length if length > 0 else 0
    is_palindrome = cell == cell[::-1]
    repeating_chars = sum(cell[i] == cell[i - 1] for i in range(1, length))
    repeating_words = sum(cell.split()[i] == cell.split()[i - 1] for i in range(1, len(cell.split())))
    first_char_type = ord(cell[0]) if length > 0 else 0
    last_char_type = ord(cell[-1]) if length > 0 else 0
    most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
    least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)
    digit_frequency = num_digits / length if length > 0 else 0
    punctuation_frequency = num_punctuation / length if length > 0 else 0
    whitespace_frequency = num_whitespace / length if length > 0 else 0
    char_diversity_ratio = unique_chars / length if length > 0 else 0
    num_alpha_sequences = sum(1 for part in cell.split() if part.isalpha())

    return [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, num_whitespace, num_special_chars,
        proportion_uppercase, proportion_digits, proportion_punctuation, proportion_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length, proportion_words,
        contains_email, contains_url, contains_hashtag, contains_at_symbol, is_numeric, is_alpha,
        is_alphanumeric, is_capitalized, shannon_entropy, unique_chars, proportion_vowels, is_palindrome,
        repeating_chars, repeating_words, first_char_type, last_char_type, most_frequent_char,
        least_frequent_char, digit_frequency, punctuation_frequency, whitespace_frequency,
        char_diversity_ratio, num_alpha_sequences
    ]

def combine_features_for_row(row, model):
    combined_row_features = []
    for cell in row:
        syntactic_features = extract_syntactic_features(cell)
        semantic_features = model.encode([str(cell)], convert_to_tensor=False)[0]
        combined_features = np.hstack([syntactic_features, semantic_features])
        combined_row_features.append(combined_features)
    return np.array(combined_row_features)



def resize_dataframe(df, target_rows=101, target_cols=50):
    try:
        print("\n🛠 Debug: Resizing DataFrame...")

        # Convert all values to numeric, replacing non-numeric with NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Flatten values and remove NaNs
        values = df_numeric.values.flatten()
        values = values[~np.isnan(values)]  # Ensure only numeric values remain

        total_values = target_rows * target_cols

        # Adjust number of values dynamically
        if len(values) > total_values:
            # Truncate excess values
            values = values[:total_values]
        elif len(values) < total_values:
            # Pad with repeated values
            extra_needed = total_values - len(values)
            repeated_values = np.tile(values, (extra_needed // len(values)) + 1)[:extra_needed]
            values = np.concatenate([values, repeated_values])

        # Create resized dataframe
        resized_df = pd.DataFrame(values.reshape(target_rows, target_cols))

        # Maintain original column names if possible
        if len(df.columns) >= target_cols:
            resized_df.columns = df.columns[:target_cols]
        else:
            # Add placeholder column names
            new_columns = list(df.columns) + [f"Extra_Feature_{i+1}" for i in range(target_cols - len(df.columns))]
            resized_df.columns = new_columns

        print(f"✅ Resized DataFrame Shape: {resized_df.shape}")
        return resized_df

    except Exception as e:
        print(f"⚠ Error during resizing: {e}")
        return df, resized_df
    
def main():
    # Load relational Table
    relational_data = pd.read_csv("California_Houses.csv")
    
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
    print(final_features[:20])
    print(f"Feature Tensor Shape: {final_features.shape}")

if __name__ == "__main__":
    main()
