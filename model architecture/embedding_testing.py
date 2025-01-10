# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize
# import random
# import string


 
    
# # Generate random synthetic data for testing
# def generate_syntactic_dataframe(num_rows=101, num_cols=50):
#     """Generate a DataFrame with random syntactic cell data."""
#     def random_string():
#         length = random.randint(1, 20)
#         return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))

#     return pd.DataFrame([[random_string() for _ in range(num_cols)] for _ in range(num_rows)])

# # Extract syntactic features for a cell
# def extract_syntactic_features(cell):
#     """Extract a fixed 39-dimensional syntactic feature vector from a cell."""
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
#     repeating_chars = sum(cell[i] == cell[i-1] for i in range(1, length))
#     repeating_words = sum(cell.split()[i] == cell.split()[i-1] for i in range(1, len(cell.split())))
#     first_char_type = ord(cell[0]) if length > 0 else 0
#     last_char_type = ord(cell[-1]) if length > 0 else 0
#     most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
#     least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)
#     digit_frequency = num_digits / length if length > 0 else 0
#     punctuation_frequency = num_punctuation / length if length > 0 else 0
#     whitespace_frequency = num_whitespace / length if length > 0 else 0
#     char_diversity_ratio = unique_chars / length if length > 0 else 0
#     num_alpha_sequences = sum(1 for part in cell.split() if part.isalpha())

#     # Return all 39 features
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

# # Combine syntactic and semantic features for a row
# def combine_features_for_row(row, model):
#     """Combine syntactic and semantic features for a single row."""
#     combined_row_features = []
#     for cell in row:
#         # Extract syntactic features
#         syntactic_features = extract_syntactic_features(cell)
#         syntactic_features = np.array(syntactic_features)  # Convert to NumPy array
#         print("Syntactic Features Shape:", syntactic_features.shape)  # Print shape
#         # Extract semantic features
#         semantic_features = model.encode([str(cell)], convert_to_tensor=False)[0]  # Single embedding
#         print("Semantic Features Shape:", semantic_features.shape) 
#         # Combine syntactic and semantic features
#         combined_features = np.hstack([syntactic_features, semantic_features])
#         combined_row_features.append(combined_features)
#     return np.array(combined_row_features)

# # Main function
# def main():
#     # Generate Random Table Data
#     df = generate_syntactic_dataframe(num_rows=101, num_cols=50)
#     print("Sample DataFrame:")
#     print(df.head())
    
#     # Load Pre-trained Sentence-BERT Model
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
#     # Extract features row by row
#     all_features = []
#     for row in df.itertuples(index=False):
#         row_features = combine_features_for_row(row, model)
#         all_features.append(row_features)
    
#     # Convert to 3D Tensor (101, 50, 423)
#     final_features = np.stack(all_features)
#     print("\nFinal Input Feature Tensor:")
#     print("Shape:", final_features.shape)
    
#     # Save as CSV for verification
#     final_features_reshaped = final_features.reshape(-1, final_features.shape[-1])  # Flatten for CSV
#     final_features_df = pd.DataFrame(final_features_reshaped)
#     final_features_df.to_csv('final_features.csv', index=False, header=False)
#     print("\nFeatures saved to 'final_features.csv'.")

# # Run the main function
# if __name__ == "__main__":
#     main()



import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import random
import string
from data_generation import generate_relational_dataframe


def unstack_dataframe(df, stack_start_idx=None, stack_end_idx=None):

    try:
        # Validate indices
        if stack_start_idx is None or stack_end_idx is None:
            raise ValueError("Both `stack_start_idx` and `stack_end_idx` must be provided.")
        
        if stack_start_idx < 0 or stack_end_idx >= len(df.columns):
            raise ValueError("Column indices are out of bounds.")
        
        if stack_start_idx > stack_end_idx:
            raise ValueError("`stack_start_idx` must be less than or equal to `stack_end_idx`.")
        
        # Separate columns into ID variables and homogeneous columns
        id_vars = df.columns[:stack_start_idx].tolist() + df.columns[stack_end_idx + 1:].tolist()
        value_vars = df.columns[stack_start_idx:stack_end_idx + 1].tolist()
        
        # Melt the DataFrame
        unstacked_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Attributes',   # Column for homogeneous column names
            value_name='Values'      # Column for corresponding values
        )
        
        return unstacked_df
    
    except Exception as e:
        print(f"Error during unstacking: {e}")
        return df
    
    
def transpose_dataframe_inverse(df):
    # Transpose the dataframe back to its original format
    try:
        transposed = df.set_index(df.columns[0]).T.reset_index()
        return transposed
    except Exception as error:
        print(f"Error during transpose inverse: {error}")
        return df
    


# Apply a random inverse operator
def apply_random_inverse_operator(df):
    operators = {
        "unstack": unstack_dataframe,
        "transpose": transpose_dataframe_inverse
    }
    operator_name = random.choice(list(operators.keys()))
    transformed_df = operators[operator_name](df)
    return transformed_df, operator_name

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
        syntactic_features = np.array(syntactic_features)  
        # print(f"Syntactic feature shape: {syntactic_features.shape}")  
        semantic_features = model.encode([str(cell)], convert_to_tensor=False)[0]  
        # print(f"Semantic feature shape: {semantic_features.shape}")  
        combined_features = np.hstack([syntactic_features, semantic_features]) 
        combined_row_features.append(combined_features)
    return np.array(combined_row_features)


def main():
    relational_data = pd.read_csv("relational_data.csv")

    # Apply a random inverse operator
    non_relational_table, operator_name  = apply_random_inverse_operator(relational_data )
    print(f"Applied random inverse operator: {operator_name}")
    print(f"Shape of transformed table: {non_relational_table.shape}")  
    if non_relational_table.empty:
        print("Transformed table is empty.")
        return

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Extract and combine features
    all_features = []
    for row in non_relational_table.itertuples(index=False):
        row_features = combine_features_for_row(row, model)
        # print(f"Row features shape: {row_features.shape}") 
        all_features.append(row_features)

    # Convert to tensor
    if all_features:
        final_features = np.stack(all_features)
        print(f"Feature Tensor Shape: {final_features.shape}")
    else:
        print("No features to stack.")

if __name__ == "__main__":
    main()
