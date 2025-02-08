# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import os
# import string
# from sentence_transformers import SentenceTransformer
# from torch.utils.data import Dataset, DataLoader


# # ---- Dataset Preparation ----
# class TableDataset(Dataset):
#     def __init__(self, folder_path):
#         self.folder_path = folder_path
#         self.files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#         self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#     def __len__(self):
#         return len(self.files)

#     def extract_features(self, df, target_shape=(101, 50, 423)):
#         df = df.astype(str)
#         all_features = []

#         for row in df.itertuples(index=False):
#             row_features = []
#             for cell in row:
#                 semantic_features = self.model.encode([cell], convert_to_tensor=False)[0]
#                 row_features.append(semantic_features)
#             all_features.append(np.hstack(row_features))

#         # Convert to NumPy and reshape correctly
#         all_features = np.array(all_features, dtype=np.float32)

#         # Ensure fixed shape (101, 50, 423)
#         target_rows, target_cols, feature_dim = target_shape
#         feature_tensor = torch.tensor(all_features, dtype=torch.float32)

#         # Fix shape mismatch
#         needed_size = target_rows * target_cols * feature_dim
#         feature_tensor = torch.flatten(feature_tensor)

#         if feature_tensor.numel() < needed_size:
#             padding_size = needed_size - feature_tensor.numel()
#             padding = torch.zeros(padding_size, dtype=torch.float32)
#             feature_tensor = torch.cat([feature_tensor, padding])
#         else:
#             feature_tensor = feature_tensor[:needed_size]  # Truncate if too large

#         return feature_tensor.view(target_rows, target_cols, feature_dim)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.folder_path, self.files[idx])
#         df = pd.read_csv(file_path)

#         # Ensure equal sampling of both operators
#         if idx % 2 == 0:
#             operator = "unstack"
#         else:
#             operator = "transpose"

#         transformed_df = df.T if operator == "transpose" else df.melt()
#         transformed_df = transformed_df.sample(n=min(101, len(transformed_df)), replace=True)
#         transformed_df = transformed_df.iloc[:, :50] if transformed_df.shape[1] > 50 else transformed_df

#         features = self.extract_features(transformed_df, target_shape=(101, 50, 423))
#         label = torch.tensor(0 if operator == "unstack" else 1, dtype=torch.long)

#         return features, label


# # ---- Model Architecture ----
# class OperatorClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super(OperatorClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 100)
#         self.dropout1 = nn.Dropout(p=0.3)  # Prevent overfitting
#         self.fc2 = nn.Linear(100, 68)
#         self.dropout2 = nn.Dropout(p=0.3)  # Additional dropout
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout1(x)  # Apply dropout
#         x = self.fc2(x)
#         x = self.dropout2(x)  # Apply dropout
#         return x


# # ---- Train Model ----
# def train_model():
#     dataset = TableDataset(folder_path="testing_tables")
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     for i in range(len(dataset)):
#         sample_features, _ = dataset[i]
#         if sample_features.shape == (101, 50, 423):
#             break

#     input_dim = sample_features.numel()

#     model = OperatorClassifier(input_dim=input_dim)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#     loss_fn = nn.CrossEntropyLoss()

#     epochs = 20  # Increase to 20 for better convergence
#     for epoch in range(epochs):
#         total_loss = 0
#         for features, labels in dataloader:
#             features = features.view(features.size(0), -1)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = loss_fn(outputs, labels)
#             print(f"Loss: {loss.item()}")
#             print(f"labels Examples: {labels}")
#             loss.backward()
#             optimizer.step()
#         scheduler.step()  # Reduce learning rate over time

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")


#     torch.save(model.state_dict(), "trained_model.pth")
#     print("✅ Model saved as trained_model.pth")

# if __name__ == "__main__":
#     train_model()




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
import os

# Define Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Function to Apply Inverse Transformations
def apply_inverse_transformation(df, transformation):
    """Applies the inverse transformation while keeping table shape stable."""
    if transformation == "unstack":
        unstacked_df = df.stack().reset_index()

        # Limit unstacked DataFrame to 50 columns
        if unstacked_df.shape[1] > 50:
            unstacked_df = unstacked_df.iloc[:, :50]
        elif unstacked_df.shape[1] < 50:
            extra_cols = pd.DataFrame(np.zeros((unstacked_df.shape[0], 50 - unstacked_df.shape[1])), 
                                      columns=[f"Extra_{i}" for i in range(50 - unstacked_df.shape[1])])
            unstacked_df = pd.concat([unstacked_df, extra_cols], axis=1)

        return unstacked_df

    elif transformation == "transpose":
        return df.T  # Transpose remains unchanged

    else:
        raise ValueError("Unsupported transformation")


# Define Dataset Class
class TableDataset(Dataset):
    def __init__(self, data_folder):
        self.files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        self.data_folder = data_folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.files[idx])
        df = pd.read_csv(file_path)

        transformation = random.choice(["unstack", "transpose"])
        transformed_df = apply_inverse_transformation(df, transformation)

        resized_table = self.resize_dataframe(transformed_df.values)

        # ✅ Normalize data to range [0,1] to avoid huge values
        resized_table = (resized_table - np.min(resized_table)) / (np.max(resized_table) - np.min(resized_table) + 1e-8)

        resized_table = np.array(resized_table, dtype=np.float32)
        label = 0 if transformation == "unstack" else 1

        return torch.tensor(resized_table, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    def resize_dataframe(self, table):
        """Ensures DataFrame is exactly (101,50) with numeric-only values."""
        target_rows, target_cols = 101, 50
        table = table[:target_rows, :target_cols]  # Trim if too large

        # Convert all values to strings
        table = np.array(table, dtype=str)

        # Encode categorical values to numeric using hash encoding
        table_numeric = np.vectorize(hash)(table).astype(np.float32)  # Convert strings to float32

        # Fill missing rows
        while table_numeric.shape[0] < target_rows:
            table_numeric = np.vstack((table_numeric, table_numeric[:(target_rows - table_numeric.shape[0])]))

        # Fill missing columns
        while table_numeric.shape[1] < target_cols:
            extra_cols = table_numeric[:, :target_cols - table_numeric.shape[1]]
            table_numeric = np.hstack((table_numeric, extra_cols))

        return table_numeric


# Define CNN Model
class TableCNN(nn.Module):
    def __init__(self):
        super(TableCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Forward pass a dummy input to find the actual size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 101, 50)
            dummy_output = self.pool(self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy_input))))))
            self.flattened_dim = dummy_output.view(1, -1).shape[1]
            print(f"Dynamically Computed FC Input Size: {self.flattened_dim}")  # Debug

        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train():
    data_folder = "testing_tables"
    dataset = TableDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TableCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print("NaN encountered. Stopping this batch.")
                exit()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader)}")

    # Save trained model
    torch.save(model.state_dict(), "trained_model.pth")
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train()
