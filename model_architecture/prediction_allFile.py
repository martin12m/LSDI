

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
    # Transpose the dataframe back to its original format
    try:
        transposed = df.set_index(df.columns[0]).T.reset_index()
        return transposed
    except Exception as error:
        print(f"Error during transpose inverse: {error}")
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

def resize_dataframe(df, target_rows=101, target_cols=50, reference_columns=None):
    try:
        original_columns = df.columns.tolist()

        values = df.values.flatten()
        values = values[~np.isnan(values)]
        values = values[values != 0]

        total_values = target_rows * target_cols

        if len(values) > total_values:
            values = values[:total_values]
        elif len(values) < total_values:
            extra_needed = total_values - len(values)
            repeated_values = np.tile(values, (extra_needed // len(values)) + 1)[:extra_needed]
            values = np.concatenate([values, repeated_values])

        resized_df = pd.DataFrame(values.reshape(target_rows, target_cols))

        if reference_columns is None:
            if len(original_columns) >= target_cols:
                resized_df.columns = original_columns[:target_cols]
            else:
                new_columns = original_columns + [f"Extra_Feature_{i+1}" for i in range(target_cols - len(original_columns))]
                resized_df.columns = new_columns
            reference_columns = resized_df.columns.tolist()
        else:
            resized_df.columns = reference_columns

        return resized_df, reference_columns

    except Exception as e:
        print(f"Error during resizing: {e}")
        return df, reference_columns

# ---- Model Layers ----
class DimensionReductionLayer(nn.Module):
    def __init__(self, in_channels=423, mid_channels=64, out_channels=32, num_oprators=4, sequence_length=None):
        super(DimensionReductionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        return x


class FeatureExtractionLayer(nn.Module):
    def __init__(self, expected_output_size=1600):
        super(FeatureExtractionLayer, self).__init__()

        # *Header Processing Path (32 → 8)*
        self.header_conv = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
        self.header_activation = nn.ReLU()

        # *Column Processing Path (32 → 8)*
        self.column_conv1 = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
        self.column_activation1 = nn.ReLU()
        self.column_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Pooling

        # *Row Processing Path (32 → 8)*
        self.row_conv1 = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
        self.row_activation1 = nn.ReLU()
        self.row_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Pooling

        # *Flattening*
        self.expected_output_size = expected_output_size
        self.flatten = nn.Flatten()

    def forward(self, x):
        print(f"Before Processing in FeatureExtractionLayer: {x.shape}")  # Debugging print

        batch_size, channels, height, width = x.shape  # Extract dimensions

        # *Separate Header and Data*
        header = x[:, :, :, 0]  # ✅ Extract first COLUMN instead of first row
        print(f"Header Shape Before Conv1D: {header.shape}")  

        header = header.permute(0, 1, 2)  # (batch, channels, width) ✅ Correct
        header = self.header_activation(self.header_conv(header))  
        print(f"Header Shape After Conv1D: {header.shape}") 
        # *Fixing the Column Shape Before Conv1D*
        
        
        # Compute width dynamically
        batch_size, channels, height, width = x.shape  

        # Permute the tensor to rearrange dimensions
        columns = x.permute(0, 3, 1, 2)  

        # Check for mismatches
        if columns.numel() != batch_size * 32 * width:
            print(f"Fixing width: Original Width {width}, Recomputed Width {columns.numel() // (batch_size * 32)}")
            width = columns.numel() // (batch_size * 32)  # Adjust width dynamically

        # Reshape with correct dimensions
        columns = columns.reshape(batch_size, 32, width)  

        # columns = x.permute(0, 3, 1, 2)
        # print(f"Column Shape Before Flattening: {columns.shape}")  # ✅ Now [1, 50, 32]

        # # *Flatten columns correctly*
        # new_width = columns.shape[1]  # ✅ Extract the correct width dynamically
        # print(f"new width: {new_width}")
        # columns = columns.reshape(batch_size, 32, width)  # ✅ Now adaptive
        print(f"Fixed Column Shape Before Conv1D: {columns.shape}")  # Debugging print

        columns = self.column_activation1(self.column_conv1(columns))
        columns = self.column_pool(columns)  # Ensure correct final size
        print(f"Column Shape After Conv1D: {columns.shape}")  # Debugging print

        rows = x.permute(0, 1, 3, 2)  # ✅ Correct ordering: (batch, height, channels, width)
        print(f"Row Shape Before Flattening: {rows.shape}")  # ✅ Now [1, 101, 32]

        # *Flatten rows correctly*
        new_height = rows.shape[1]  # ✅ Extract the correct height dynamically
        print(f"new height: {new_height}")
        rows = rows.reshape(batch_size, 32, new_height)  # ✅ Now adaptive
        print(f"Fixed Row Shape Before Conv1D: {rows.shape}")  # Debugging print

        rows = self.row_activation1(self.row_conv1(rows))
        rows = self.row_pool(rows)  # Ensure correct final size

        # *Concatenate Header with Columns & Rows*
        combined_features = torch.cat([header, columns, rows], dim=2)  # Header + Columns + Rows
        print(f"Combined Features Shape Before Flattening: {combined_features.shape}")  # Debugging print

        # *Flatten Before Output*
        output_features = self.flatten(combined_features)
        print(f"Extracted Features Shape: {output_features.shape}")  # ✅ Expect [1, 1600]

        return output_features


class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=270):
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

# ---- Main Prediction Pipeline ----
def main():
    # Generate relational table

    relational_data = pd.read_csv("California_Houses.csv")
    # non_relational_table, operator_name  = apply_random_inverse_operator(relational_data)
    # print(f"Applied random inverse operator: {operator_name}")
    
    non_relational_table = unstack_dataframe(relational_data)
    non_relational_table.to_csv("non_relational_data.csv", index=False)
    print(non_relational_table.shape) 
    print(f"Shape of transformed table: {non_relational_table.shape}") 
    if non_relational_table.empty:
        print("Transformed table is empty. Exiting.")
        return

    resized_df, saved_columns = resize_dataframe(non_relational_table)
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

    # Dimension Reduction Layer
    input_tensor = torch.rand(final_features.shape)
    n, m, d = input_tensor.shape  # (n, m, d)
    input_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0)
    input_tensor_reshaped = input_tensor.permute(0, 3, 1, 2)
    print(f"Input Tensor Shape: {input_tensor_reshaped.shape}")
    batch_size, in_channels, height, width = input_tensor_reshaped.shape
    dim_reduction_layer = DimensionReductionLayer(in_channels=in_channels, mid_channels=64, out_channels=32)
    output_tensor = dim_reduction_layer(input_tensor_reshaped)
    print(f"Output Tensor Shape: {output_tensor.shape}")
    
    #Feature Extraction Layer
    # input_reduced_tensor = torch.rand(output_tensor.shape)
    input_reduced_tensor = output_tensor.permute(0, 1, 3, 2)
    print(f"Input Reduced Tensor Shape: {input_reduced_tensor.shape}")
    feature_extraction = FeatureExtractionLayer()
    features = feature_extraction(input_reduced_tensor)
    print("Extracted Features Shape:", features.shape)


if __name__ == "__main__":
    main()
