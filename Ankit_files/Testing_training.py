# import os
# import time
# import torch
# import pandas as pd
# import numpy as np
# import random
# import string
# from sentence_transformers import SentenceTransformer

# # ---- Configuration ----
# INPUT_FOLDER = "training_data"  # Path to stored tables
# TARGET_ROWS = 101
# TARGET_COLS = 50
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---- Operators and Inverse Operators ----
# def unstack_dataframe(df):
#     """
#     Implements the Unstack Operator (Inverse of Stack).
#     Transforms homogeneous columns into rows.
#     """
#     try:
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#         numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        
#         if len(categorical_cols) < 1 or len(numeric_cols) == 0:
#             raise ValueError("Table must contain categorical and numeric columns for unstacking.")
        
#         pivot_column = categorical_cols[-1]  # Last categorical column as pivot
#         index_cols = categorical_cols[:-1]

#         unstacked_df = df.pivot(index=index_cols, columns=pivot_column, values=numeric_cols[0])
#         return unstacked_df.reset_index()
    
#     except Exception as e:
#         print(f"❌ Error during unstacking: {e}")
#         return df

# def transpose_dataframe_inverse(df):
#     """
#     Transposes a DataFrame while maintaining semantic meaning.
#     """
#     try:
#         df.columns = [str(col) for col in df.columns]
#         transposed_df = df.set_index(df.columns[0]).T.reset_index()
#         return transposed_df
    
#     except Exception as e:
#         print(f"⚠ Error during transpose: {e}")
#         return df

# def apply_random_inverse_operator(df):
#     """
#     Randomly applies an inverse operator (either Unstack or Transpose).
#     """
#     operators = {
#         "unstack": unstack_dataframe,
#         "transpose": transpose_dataframe_inverse
#     }
#     operator_name = random.choice(list(operators.keys()))
#     transformed_df = operators[operator_name](df)
#     return transformed_df, operator_name

# # ---- Feature Extraction ----
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

# # ---- Resize Table ----
# def resize_dataframe(df, target_rows=TARGET_ROWS, target_cols=TARGET_COLS):
#     """
#     Ensures table has a consistent size (101 rows × 50 cols).
#     ✅ Does NOT use excessive padding.
#     ✅ Ensures homogeneity is preserved.
#     """
#     try:
#         num_rows, num_cols = df.shape

#         # If too many rows, truncate
#         if num_rows > target_rows:
#             df = df.iloc[:target_rows, :]
#         # If too few rows, duplicate rows until target size is met
#         elif num_rows < target_rows:
#             repeat_factor = (target_rows // num_rows) + 1
#             df = pd.concat([df] * repeat_factor, ignore_index=True).iloc[:target_rows, :]

#         # If too many columns, truncate
#         if num_cols > target_cols:
#             df = df.iloc[:, :target_cols]
#         # If too few columns, duplicate columns until target size is met
#         elif num_cols < target_cols:
#             repeat_factor = (target_cols // num_cols) + 1
#             df = pd.concat([df] * repeat_factor, axis=1).iloc[:, :target_cols]

#         return df
    
#     except Exception as e:
#         print(f"⚠ Error during resizing: {e}")
#         return df

# # ---- Batch Embedding Optimization ----
# def extract_batch_embeddings(df, model, batch_size=64):
#     """
#     Efficiently extracts sentence embeddings using batch processing.
#     🚀 10x Faster: Uses GPU for batch encoding instead of cell-by-cell.
#     """
#     all_text = [str(cell) if isinstance(cell, str) else '' for row in df.itertuples(index=False) for cell in row]

#     # Perform batch encoding (GPU optimized)
#     embeddings = model.encode(all_text, batch_size=batch_size, device=DEVICE, convert_to_tensor=False)

#     # Reshape back to (rows, columns, embedding_dim)
#     return embeddings.reshape(df.shape[0], df.shape[1], -1)

# # ---- Process Single File ----
# def process_file(file_path, model):
#     """
#     Loads a CSV file, applies transformations, resizes, and extracts embeddings.
#     """
#     try:
#         print(f"\n🔍 Processing File: {file_path}")

#         # Load table
#         relational_data = pd.read_csv(file_path)

#         # Apply inverse transformation
#         start_time = time.time()
#         transformed_table, operator_name = apply_random_inverse_operator(relational_data)
#         transform_time = time.time() - start_time
#         print(f"✅ Applied {operator_name} in {transform_time:.2f} seconds")

#         # Resize table
#         start_time = time.time()
#         resized_df = resize_dataframe(transformed_table)
#         resize_time = time.time() - start_time
#         print(f"✅ Resized Table: {resized_df.shape} in {resize_time:.2f} seconds")

#         # Feature extraction (measure time separately)
#         embedding_time_start = time.time()
#         embeddings = extract_batch_embeddings(resized_df, model, batch_size=64)
#         embedding_time_end = time.time() - embedding_time_start
#         print(f"🚀 Embedding Layer Time: {embedding_time_end:.2f} seconds")

#         # Save features
#         np.save(f"features_{os.path.basename(file_path).replace('.csv', '.npy')}", embeddings)
#         print(f"✅ Saved Embeddings for {file_path}")

#     except Exception as e:
#         print(f"❌ Error processing file {file_path}: {e}")

# # ---- Main Function ----
# def main():
#     """
#     Loads and processes all tables in the dataset, timing only embedding.
#     """
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)

#     input_folder = INPUT_FOLDER
#     total_files = len([f for f in os.listdir(input_folder) if f.endswith('.csv')])
    
#     print(f"\n🚀 Starting Embedding Benchmark on {total_files} Tables...\n")

#     total_time_start = time.time()
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(input_folder, filename)
#             process_file(file_path, model)
#     total_time_end = time.time() - total_time_start

#     print(f"\n✅ Finished Processing {total_files} Tables in {total_time_end:.2f} seconds.")

# # ---- Run the Script ----
# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ModelA(nn.Module):
    def __init__(self, in_channels=423, mid_channels=64, out_channels=32):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        return x 


class ModelB(nn.Module):
    def __init__(self, in_channels=32, mid_channels=64, out_channels=8):
        super(ModelB, self).__init__()

        # ==== Column Feature Extraction ====
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))  # Pool entire column

        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        # ==== Row Feature Extraction ====
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))  # Pool entire row

        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))

        # ==== Header Feature Extraction ====
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # ====== Header Extraction ======
        header = x[:, :, 0:1, :]  # Extract first row
        x = x[:, :, 1:, :]  # Remaining rows
        header_features = self.activation(self.conv1x1_header(header))

        # ====== Column Feature Extraction ======
        global_col1 = self.activation(self.conv1x1_col1(x))
        local_col1 = self.activation(self.conv1x2_col1(x))

        min_width = min(global_col1.shape[3], local_col1.shape[3])
        global_col1 = global_col1[:, :, :, :min_width]
        local_col1 = local_col1[:, :, :, :min_width]

        global_col1_pooled = self.avgpool_col1(global_col1)
        local_col1_pooled = self.avgpool_col1(local_col1)

        column_features1 = global_col1_pooled + local_col1_pooled

        global_col2 = self.activation(self.conv1x1_col2(column_features1))
        local_col2 = self.activation(self.conv1x2_col2(column_features1))

        min_width2 = min(global_col2.shape[3], local_col2.shape[3])
        global_col2 = global_col2[:, :, :, :min_width2]
        local_col2 = local_col2[:, :, :, :min_width2]

        column_features2 = global_col2 + local_col2  # Final column shape = (8, 50)

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

        row_features2 = global_row2 + local_row2  # Final row shape = (8, 100)

        # ====== Flatten and Combine Features ======
        column_features_flat = column_features2.reshape(batch_size, -1)  # (1, 400)
        row_features_flat = row_features2.reshape(batch_size, -1)  # (1, 800)
        header_features_flat = header_features.reshape(batch_size, -1)  # (1, 400)

        combined_features = torch.cat([column_features_flat, row_features_flat, header_features_flat], dim=1)

        return combined_features  


class ModelC(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(ModelC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x  


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.modelA = ModelA(in_channels=423, mid_channels=64, out_channels=32)  # Dimension Reduction
        self.modelB = ModelB(in_channels=32, mid_channels=64, out_channels=8)  # Feature Extraction
        self.modelC = ModelC(input_dim=1600, output_dim=2)  # Output Layer

    def forward(self, x):
        x = self.modelA(x) 
        print(f"Dimenstion Reduction layer Output Shape: {x.shape}")
        x = self.modelB(x)  
        print(f"Feature Extraction layer Output Shape: {x.shape}")
        x = self.modelC(x) 
        print(f"Classification layer Output Shape: {x.shape}") 
        return x


if __name__ == "__main__":
    sample_input = torch.randn(1, 423, 101, 50)  # Simulating a table input
    model = FullModel()
    output = model(sample_input)
    print(f"🔍 Output Shape: {output.shape}") 



# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset

# # ==========================
# # 🔹 Model A: Dimension Reduction Layer
# # ==========================
# class ModelA(nn.Module):
#     def __init__(self, in_channels=423, mid_channels=64, out_channels=32):
#         super(ModelA, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
#         self.batch_norm1 = nn.BatchNorm2d(mid_channels)
#         self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         x = self.activation(self.batch_norm1(self.conv1(x)))
#         x = self.activation(self.batch_norm2(self.conv2(x)))
#         return x  # Output Shape: (batch_size, 32, height, width)

# # ==========================
# # 🔹 Model B: Feature Extraction Layer
# # ==========================
# class ModelB(nn.Module):
#     def __init__(self, in_channels=32, mid_channels=64, out_channels=8):
#         super(ModelB, self).__init__()

#         # ==== Column Feature Extraction ====
#         self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
#         self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
#         self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))  # Pool entire column

#         self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
#         self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

#         # ==== Row Feature Extraction ====
#         self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
#         self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
#         self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))  # Pool entire row

#         self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
#         self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))

#         # ==== Header Feature Extraction ====
#         self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

#         self.activation = nn.ReLU()

#     def forward(self, x):
#         batch_size, _, height, width = x.shape

#         # ====== Header Extraction ======
#         header = x[:, :, 0:1, :]  # Extract first row
#         x = x[:, :, 1:, :]  # Remaining rows
#         header_features = self.activation(self.conv1x1_header(header))

#         # ====== Column Feature Extraction ======
#         global_col1 = self.activation(self.conv1x1_col1(x))
#         local_col1 = self.activation(self.conv1x2_col1(x))

#         min_width = min(global_col1.shape[3], local_col1.shape[3])
#         global_col1 = global_col1[:, :, :, :min_width]
#         local_col1 = local_col1[:, :, :, :min_width]

#         global_col1_pooled = self.avgpool_col1(global_col1)
#         local_col1_pooled = self.avgpool_col1(local_col1)

#         column_features1 = global_col1_pooled + local_col1_pooled

#         global_col2 = self.activation(self.conv1x1_col2(column_features1))
#         local_col2 = self.activation(self.conv1x2_col2(column_features1))

#         min_width2 = min(global_col2.shape[3], local_col2.shape[3])
#         global_col2 = global_col2[:, :, :, :min_width2]
#         local_col2 = local_col2[:, :, :, :min_width2]

#         column_features2 = global_col2 + local_col2  # Final column shape = (8, 50)

#         # ====== Row Feature Extraction ======
#         global_row1 = self.activation(self.conv1x1_row1(x))
#         local_row1 = self.activation(self.conv1x2_row1(x))

#         min_height = min(global_row1.shape[2], local_row1.shape[2])
#         global_row1 = global_row1[:, :, :min_height, :]
#         local_row1 = local_row1[:, :, :min_height, :]

#         global_row1_pooled = self.avgpool_row1(global_row1)
#         local_row1_pooled = self.avgpool_row1(local_row1)

#         row_features1 = global_row1_pooled + local_row1_pooled

#         global_row2 = self.activation(self.conv1x1_row2(row_features1))
#         local_row2 = self.activation(self.conv1x2_row2(row_features1))

#         min_height2 = min(global_row2.shape[2], local_row2.shape[2])
#         global_row2 = global_row2[:, :, :min_height2, :]
#         local_row2 = local_row2[:, :, :min_height2, :]

#         row_features2 = global_row2 + local_row2  # Final row shape = (8, 100)

#         # ====== Flatten and Combine Features ======
#         column_features_flat = column_features2.reshape(batch_size, -1)  # (1, 400)
#         row_features_flat = row_features2.reshape(batch_size, -1)  # (1, 800)
#         header_features_flat = header_features.reshape(batch_size, -1)  # (1, 400)

#         combined_features = torch.cat([column_features_flat, row_features_flat, header_features_flat], dim=1)

#         return combined_features  # Shape: (1, 1600)

# # ==========================
# # 🔹 Model C: Output Layer (Classification)
# # ==========================
# class ModelC(nn.Module):
#     def __init__(self, input_dim, output_dim=2):
#         super(ModelC, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.activation = nn.ReLU()
#         self.fc2 = nn.Linear(512, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         return x  # Final logits for classification

# # ==========================
# # 🔹 Full Model (Pipeline)
# # ==========================
# class FullModel(nn.Module):
#     def __init__(self):
#         super(FullModel, self).__init__()
#         self.modelA = ModelA(in_channels=423, mid_channels=64, out_channels=32)
#         self.modelB = ModelB(in_channels=32, mid_channels=64, out_channels=8)
#         self.modelC = ModelC(input_dim=1600, output_dim=2)

#     def forward(self, x):
#         x = self.modelA(x)
#         x = self.modelB(x)
#         x = self.modelC(x)
#         return x

# # ==========================
# # 🔹 Training Pipeline
# # ==========================
# def train_model():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load precomputed embeddings
#     feature_files = [f for f in os.listdir() if f.startswith("features_") and f.endswith(".npy")]
#     label_files = [f.replace("features_", "labels_") for f in feature_files]

#     # Load data
#     X = np.vstack([np.load(f) for f in feature_files])
#     y = np.hstack([np.load(f) for f in label_files])

#     # Convert to PyTorch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
#     y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)

#     dataset = TensorDataset(X_tensor, y_tensor)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Model setup
#     model = FullModel().to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(10):
#         for batch_X, batch_y in dataloader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# if __name__ == "__main__":
#     train_model()
