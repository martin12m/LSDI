


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# import numpy as np
# import random
# import os

# # Define Constants
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 20
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define Function to Apply Inverse Transformations
# def apply_inverse_transformation(df, transformation):
#     """Applies the inverse transformation while keeping table shape stable."""
#     if transformation == "unstack":
#         unstacked_df = df.stack().reset_index()

#         # Limit unstacked DataFrame to 50 columns
#         if unstacked_df.shape[1] > 50:
#             unstacked_df = unstacked_df.iloc[:, :50]
#         elif unstacked_df.shape[1] < 50:
#             extra_cols = pd.DataFrame(np.zeros((unstacked_df.shape[0], 50 - unstacked_df.shape[1])), 
#                                       columns=[f"Extra_{i}" for i in range(50 - unstacked_df.shape[1])])
#             unstacked_df = pd.concat([unstacked_df, extra_cols], axis=1)

#         return unstacked_df

#     elif transformation == "transpose":
#         return df.T  # Transpose remains unchanged

#     else:
#         raise ValueError("Unsupported transformation")


# # Define Dataset Class
# class TableDataset(Dataset):
#     def __init__(self, data_folder):
#         self.files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
#         self.data_folder = data_folder

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.data_folder, self.files[idx])
#         df = pd.read_csv(file_path)

#         transformation = random.choice(["unstack", "transpose"])
#         transformed_df = apply_inverse_transformation(df, transformation)

#         resized_table = self.resize_dataframe(transformed_df.values)

#         # ✅ Normalize data to range [0,1] to avoid huge values
#         resized_table = (resized_table - np.min(resized_table)) / (np.max(resized_table) - np.min(resized_table) + 1e-8)

#         resized_table = np.array(resized_table, dtype=np.float32)
#         label = 0 if transformation == "unstack" else 1

#         return torch.tensor(resized_table, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
#     def resize_dataframe(self, table):
#         """Ensures DataFrame is exactly (101,50) with numeric-only values."""
#         target_rows, target_cols = 101, 50
#         table = table[:target_rows, :target_cols]  # Trim if too large

#         # Convert all values to strings
#         table = np.array(table, dtype=str)

#         # Encode categorical values to numeric using hash encoding
#         table_numeric = np.vectorize(hash)(table).astype(np.float32)  # Convert strings to float32

#         # Fill missing rows
#         while table_numeric.shape[0] < target_rows:
#             table_numeric = np.vstack((table_numeric, table_numeric[:(target_rows - table_numeric.shape[0])]))

#         # Fill missing columns
#         while table_numeric.shape[1] < target_cols:
#             extra_cols = table_numeric[:, :target_cols - table_numeric.shape[1]]
#             table_numeric = np.hstack((table_numeric, extra_cols))

#         return table_numeric


# # Define CNN Model
# class TableCNN(nn.Module):
#     def __init__(self):
#         super(TableCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
#         # Forward pass a dummy input to find the actual size
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 1, 101, 50)
#             dummy_output = self.pool(self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy_input))))))
#             self.flattened_dim = dummy_output.view(1, -1).shape[1]
#             print(f"Dynamically Computed FC Input Size: {self.flattened_dim}")  # Debug

#         self.fc1 = nn.Linear(self.flattened_dim, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))

#         x = x.view(x.size(0), -1)  # Flatten dynamically
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # Training function
# def train():
#     data_folder = "testing_tables"
#     dataset = TableDataset(data_folder)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     model = TableCNN().to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     for epoch in range(EPOCHS):
#         running_loss = 0.0
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             if torch.isnan(loss):
#                 print("NaN encountered. Stopping this batch.")
#                 exit()
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#             optimizer.step()
#             running_loss += loss.item()

#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader)}")

#     # Save trained model
#     torch.save(model.state_dict(), "trained_model.pth")
#     print("✅ Model saved successfully!")

# if __name__ == "__main__":
#     train()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------
# Device Definition
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Pipeline Layers
# ---------------------------
class DimensionReductionLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=32):
        super(DimensionReductionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        return x

class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=8):
        """
        Note: Originally the pooling kernel was set to (1, 6) to match 6 columns.
        Here we update it to (1, 50) because we assume 50 columns.
        """
        super(FeatureExtractionLayer, self).__init__()
        # Header branch: extract features from the first row
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        # Column branch
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))
        # Row branch
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
        # Update pooling kernel to match 50 columns
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))
        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        # x shape: (batch, channels, rows, cols)
        # Extract header from the first row and data from the remaining 50 rows.
        header = x[:, :, 0:1, :]
        x_data = x[:, :, 1:, :]
        header_features = self.activation(self.conv1x1_header(header))

        # --- Column Branch ---
        global_col1 = self.activation(self.conv1x1_col1(x_data))
        local_col1 = self.activation(self.conv1x2_col1(x_data))
        # Ensure matching width
        min_width = min(global_col1.size(3), local_col1.size(3))
        global_col1 = global_col1[:, :, :, :min_width]
        local_col1 = local_col1[:, :, :, :min_width]
        # Pool along the row dimension (pool over all rows in x_data)
        pool_kernel_col = (x_data.size(2), 1)
        global_col1_pooled = F.avg_pool2d(global_col1, kernel_size=pool_kernel_col)
        local_col1_pooled = F.avg_pool2d(local_col1, kernel_size=pool_kernel_col)
        col_features1 = global_col1_pooled + local_col1_pooled

        global_col2 = self.activation(self.conv1x1_col2(col_features1))
        local_col2 = self.activation(self.conv1x2_col2(col_features1))
        min_width2 = min(global_col2.size(3), local_col2.size(3))
        global_col2 = global_col2[:, :, :, :min_width2]
        local_col2 = local_col2[:, :, :, :min_width2]
        col_features2 = global_col2 + local_col2

        # --- Row Branch ---
        global_row1 = self.activation(self.conv1x1_row1(x_data))
        local_row1 = self.activation(self.conv1x2_row1(x_data))
        min_height = min(global_row1.size(2), local_row1.size(2))
        global_row1 = global_row1[:, :, :min_height, :]
        local_row1 = local_row1[:, :, :min_height, :]
        row_global_pooled = self.avgpool_row1(global_row1)
        row_local_pooled = self.avgpool_row1(local_row1)
        row_features1 = row_global_pooled + row_local_pooled

        global_row2 = self.activation(self.conv1x1_row2(row_features1))
        local_row2 = self.activation(self.conv1x2_row2(row_features1))
        min_height2 = min(global_row2.size(2), local_row2.size(2))
        global_row2 = global_row2[:, :, :min_height2, :]
        local_row2 = local_row2[:, :, :min_height2, :]
        row_features2 = global_row2 + local_row2

        # Flatten each branch and concatenate
        col_features_flat = col_features2.reshape(batch_size, -1)
        row_features_flat = row_features2.reshape(batch_size, -1)
        header_features_flat = header_features.reshape(batch_size, -1)
        combined_features = torch.cat([col_features_flat, row_features_flat, header_features_flat], dim=1)
        return combined_features

class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class FullPipeline(nn.Module):
    def __init__(self, in_channels):
        """
        The pipeline first reduces the input dimensions, then extracts features
        and finally produces a 2-class output.
        """
        super(FullPipeline, self).__init__()
        self.dim_reduction = DimensionReductionLayer(in_channels=in_channels, mid_channels=64, out_channels=32)
        self.feature_extraction = FeatureExtractionLayer(in_channels=32, mid_channels=64, out_channels=8)
        # Create a dummy input to determine the flattened feature dimension.
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 51, 50)
            reduced = self.dim_reduction(dummy_input)
            features = self.feature_extraction(reduced)
            flat_dim = features.view(features.size(0), -1).shape[1]
        self.output_layer = OutputLayer(flat_dim, output_dim=2)

    def forward(self, x):
        x = self.dim_reduction(x)
        features = self.feature_extraction(x)
        out = self.output_layer(features)
        return out

# ---------------------------
# Training and Testing
# ---------------------------
def train_and_test():
    # --- Create Dummy Data ---
    # For this example, we assume:
    # - Each sample is of shape: (in_channels, 51, 50)
    # - in_channels might be 407 (e.g., 23 syntactic features + 384 semantic features)
    num_samples = 100
    in_channels = 407  # Adjust as needed for your feature dimension
    # Generate random features and random binary labels
    dummy_features = torch.randn(num_samples, in_channels, 51, 50)
    dummy_labels = torch.randint(0, 2, (num_samples,))
    
    # Wrap the data in a TensorDataset and split into train and test sets.
    dataset = TensorDataset(dummy_features, dummy_labels)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Create Model, Optimizer, and Loss Function ---
    model = FullPipeline(in_channels=in_channels).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # --- Testing Phase ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_and_test()
