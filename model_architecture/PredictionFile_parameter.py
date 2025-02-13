import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import string
from sentence_transformers import SentenceTransformer


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
        print("\n Debug: Transposing DataFrame...")

        # Ensure first column is categorical (Avoid numeric index issues)
        if df.dtypes[0] != 'object':
            df.columns = [f"Column_{i}" for i in range(df.shape[1])]

        # Transpose and reset index
        transposed_df = df.set_index(df.columns[0]).T.reset_index()

        # Print debug info
        print(f"Transposed DataFrame Shape: {transposed_df.shape}")
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
        print("\n Debug: Resizing DataFrame...")

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
    def __init__(self, in_channels=32, mid_channels=64, out_channels=8):
        super(FeatureExtractionLayer, self).__init__()

        # ==== Column Feature Extraction ====
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        print(f"Conv1x1_col1 Shape: {self.conv1x1_col1}")
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        print(f"Conv1x2_col1 Shape: {self.conv1x2_col1}")
        self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1)) 
        print(f"AvgPool_col1 Shape: {self.avgpool_col1}")

        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        # ==== Row Feature Extraction ====
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50)) 

        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))

        # ==== Header Feature Extraction ====
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # ====== Header Extraction ======
        header = x[:, :, 0:1, :]  
        x = x[:, :, 1:, :]
        header_features = self.activation(self.conv1x1_header(header)) 
        print(f"Header Shape: {header_features.shape}")

        # ====== Column Feature Extraction ======
        global_col1 = self.activation(self.conv1x1_col1(x))
        print(f"global Column Shape: {global_col1.shape}")  
        local_col1 = self.activation(self.conv1x2_col1(x))   
        print(f"local Column Shape: {local_col1.shape}")

    
        min_width = min(global_col1.shape[3], local_col1.shape[3])
        print(f"Min Width: {min_width}")
        global_col1 = global_col1[:, :, :, :min_width]
        print(f"Global Column Shape: {global_col1.shape}")
        local_col1 = local_col1[:, :, :, :min_width]
        print(f"Local Column Shape: {local_col1.shape}")

        global_col1_pooled = self.avgpool_col1(global_col1)  
        local_col1_pooled = self.avgpool_col1(local_col1)    

        column_features1 = global_col1_pooled + local_col1_pooled  

        global_col2 = self.activation(self.conv1x1_col2(column_features1))  
        local_col2 = self.activation(self.conv1x2_col2(column_features1))   

        min_width2 = min(global_col2.shape[3], local_col2.shape[3])
        global_col2 = global_col2[:, :, :, :min_width2]
        local_col2 = local_col2[:, :, :, :min_width2]

        column_features2 = global_col2 + local_col2  
        print(f"Column Features Shape: {column_features2.shape}")

        # ====== Row Feature Extraction ======
        global_row1 = self.activation(self.conv1x1_row1(x))  
        local_row1 = self.activation(self.conv1x2_row1(x))   


        min_height = min(global_row1.shape[2], local_row1.shape[2])
        global_row1 = global_row1[:, :, :min_height, :]
        local_row1 = local_row1[:, :, :min_height, :]

        global_row1_pooled = self.avgpool_row1(global_row1)  
        local_row1_pooled = self.avgpool_row1(local_row1)    

        row_features1 = global_row1_pooled + local_row1_pooled  

        global_row2 = self.activation(self.conv1x1_row2(row_features1))  
        local_row2 = self.activation(self.conv1x2_row2(row_features1))   

        min_height2 = min(global_row2.shape[2], local_row2.shape[2])
        global_row2 = global_row2[:, :, :min_height2, :]
        local_row2 = local_row2[:, :, :min_height2, :]

        row_features2 = global_row2 + local_row2 
        print(f"Row Features Shape: {row_features2.shape}")

        # ====== Flatten and Combine Features ======
        column_features_flat = column_features2.reshape(batch_size, -1)  
        print(f"Column Features Shape: {column_features_flat.shape}")
        row_features_flat = row_features2.reshape(batch_size, -1)       
        print(f"row_features_flat.shape", row_features_flat.shape)
        header_features_flat = header_features.reshape(batch_size, -1)  
        print(f"Header Features Shape : {header_features_flat.shape}")

        combined_features = torch.cat([column_features_flat, row_features_flat, header_features_flat], dim=1)

        return combined_features


class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2): 
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.fc2 = nn.Linear(512, output_dim)  
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x) 
        x = self.fc2(x)
        print(f"Output Layer Shape: {x.shape}")  
        return x

class ParameterPredictionLayer(nn.Module):
    def __init__(self, input_dim, num_columns, num_rows):

        super(ParameterPredictionLayer, self).__init__()
        self.fc_column = nn.Linear(input_dim, num_columns)  
        self.fc_row = nn.Linear(input_dim, num_rows) 

    def forward(self, x):
        pivot_column_scores = self.fc_column(x)  
        header_row_scores = self.fc_row(x) 
        return pivot_column_scores, header_row_scores
    
    
def main():

    relational_data = pd.read_csv("e-commerce_1.csv")
    
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
    final_features = np.stack(all_features)
    print(f"Feature Tensor Shape: {final_features.shape}")

    # Dimension Reduction Layer
    input_tensor = torch.rand(final_features.shape)
    n, m, d = input_tensor.shape 
    input_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0)
    input_tensor_reshaped = input_tensor.permute(0, 3, 1, 2)
    print(f"Input Tensor Shape: {input_tensor_reshaped.shape}")
    batch_size, in_channels, height, width = input_tensor_reshaped.shape
    dim_reduction_layer = DimensionReductionLayer(in_channels=in_channels, mid_channels=64, out_channels=32)
    output_tensor = dim_reduction_layer(input_tensor_reshaped)
    print(f"Output Tensor Shape: {output_tensor.shape}")
    
    #Feature Extraction Layer
    input_reduced_tensor = torch.rand(output_tensor.shape)
    input_reduced_tensor = output_tensor.permute(0, 1, 2, 3)
    print(f"Input Reduced Tensor Shape: {input_reduced_tensor.shape}")
    feature_extractor = FeatureExtractionLayer()
    output = feature_extractor(input_reduced_tensor)
    print(" Final Output Shape:", output.shape)  
    

    input_dim = output.shape[1]  
    print(f"Input Dimension for Output Layer: {input_dim}")
    output_layer = OutputLayer(input_dim=input_dim, output_dim=2)
    logits = output_layer(output)


    # *Convert logits to probabilities using softmax*
    operator_probs = torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()
    print(f"Softmax probabilities: {operator_probs}")


    operators = ["unstack", "transpose"]
    predicted_operator_idx = torch.argmax(logits, dim=1).tolist()
    predicted_operator = [operators[i] for i in predicted_operator_idx]
    print(f"Predicted Operator: {predicted_operator}")
    print(f"Actual operator: {operator_name}")
    
    parameter_layer = ParameterPredictionLayer(input_dim=input_dim, num_columns=50, num_rows=101)
    pivot_column_scores, header_row_scores = parameter_layer(output)
    print(f"Pivot Column Scores Shape: {pivot_column_scores.shape}")
    print(f"Header Row Scores Shape: {header_row_scores.shape}")
    predicted_pivot_column = torch.argmax(pivot_column_scores, dim=1).tolist()
    print(f"Predicted Pivot Column: {predicted_pivot_column}")
    predicted_header_row = torch.argmax(header_row_scores, dim=1).tolist()
    print(f"Predicted Header Row: {predicted_header_row}")
    

if __name__ == "__main__":
    main()