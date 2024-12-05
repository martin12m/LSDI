import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# Paths to data
relational_folder = "relational_tables_csv"
non_relational_folder = "transformed_datasets"
labels_path = "labels.csv"

class TableDataset(Dataset):
    def __init__(self, relational_folder, non_relational_folder, labels_path, target_size=1024):
        self.relational_folder = relational_folder
        self.non_relational_folder = non_relational_folder
        self.labels = pd.read_csv(labels_path)
        self.table_ids = self.labels["table_id"].values
        self.transformations = self.labels["transformation"].values
        self.target_size = target_size  # Desired size of flattened tensors

        # Map transformations to numerical labels
        self.transformation_mapping = {t: i for i, t in enumerate(self.labels["transformation"].unique())}

    def preprocess_table(self, df):
        """Convert the table to numeric format and ensure consistent size."""
        # Drop non-numeric columns and replace NaN with 0
        df = df.select_dtypes(include=[np.number]).fillna(0)

        # Flatten the table and resize to target size
        flat_array = df.to_numpy(dtype=np.float32).flatten()
        if len(flat_array) > self.target_size:
            flat_array = flat_array[:self.target_size]  # Truncate if too large
        else:
            # Pad with zeros if too small
            flat_array = np.pad(flat_array, (0, self.target_size - len(flat_array)), mode="constant")
        
        return flat_array

    def __len__(self):
        return len(self.table_ids)

    def __getitem__(self, idx):
        table_id = self.table_ids[idx]
        relational_path = os.path.join(self.relational_folder, f"{table_id}.csv")
        non_relational_path = os.path.join(self.non_relational_folder, f"non_relational_{table_id}.csv")

        # Load relational and non-relational tables
        try:
            relational = pd.read_csv(relational_path)
            non_relational = pd.read_csv(non_relational_path)
        except Exception as e:
            print(f"Error loading table {table_id}: {e}")
            return None

        # Preprocess tables
        relational_tensor = torch.tensor(self.preprocess_table(relational), dtype=torch.float32)
        non_relational_tensor = torch.tensor(self.preprocess_table(non_relational), dtype=torch.float32)

        # Label as the transformation index
        label = self.transformation_mapping[self.transformations[idx]]

        return non_relational_tensor, relational_tensor, torch.tensor(label, dtype=torch.long)




# Create dataset and DataLoader
dataset = TableDataset(relational_folder, non_relational_folder, labels_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class TransformationModel(nn.Module):
    def __init__(self, num_classes):
        super(TransformationModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers for transformation classification
        self.fc1 = nn.Linear(64 * 50, 128)  # Adjust dimensions based on flattened size
        self.fc2 = nn.Linear(128, num_classes)

        # Fully connected layers for parameter prediction
        self.param_fc1 = nn.Linear(64 * 50, 64)
        self.param_fc2 = nn.Linear(64, 2)  # Predict two parameters

    def forward(self, x):
        # Reshape input for CNN (batch_size x channels x features)
        x = x.unsqueeze(1)

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Transformation classification
        operator_out = F.relu(self.fc1(x))
        operator_out = self.fc2(operator_out)

        # Parameter prediction
        param_out = F.relu(self.param_fc1(x))
        param_out = self.param_fc2(param_out)

        return operator_out, param_out

# Initialize model
num_classes = len(dataset.labels["transformation"].unique())
model = TransformationModel(num_classes)

# Define loss functions
criterion_operator = nn.CrossEntropyLoss()
criterion_params = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for non_relational, relational, labels in dataloader:
        # Forward pass
        operator_out, param_out = model(non_relational)

        # Compute losses
        loss_operator = criterion_operator(operator_out, labels)
        loss_params = criterion_params(param_out, relational)

        # Combine losses
        loss = loss_operator + loss_params

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(operator_out, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Print epoch metrics
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "transformation_model.pth")
print("Model saved successfully!")

