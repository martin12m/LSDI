import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


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

        # Column feature extraction
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))  # Pool entire column

        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        # Row feature extraction
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))  # Pool entire row

        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))

        # Header feature extraction
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Header extraction
        header = x[:, :, 0:1, :]
        x = x[:, :, 1:, :]

        header_features_1 = self.activation(self.conv1x1_header(header))
        header_features_2 = self.activation(self.conv1x2_header(header))

        min_width_header = min(header_features_1.shape[3], header_features_2.shape[3])
        header_features_1 = header_features_1[:, :, :, :min_width_header]
        header_features_2 = header_features_2[:, :, :, :min_width_header]

        header = header_features_1 + header_features_2

        # Column feature extraction
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

        column_features2 = global_col2 + local_col2

        # Row feature extraction
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

        # Flatten and combine features
        column_features_flat = column_features2.reshape(batch_size, -1)
        row_features_flat = row_features2.reshape(batch_size, -1)
        header_features_flat = header.reshape(batch_size, -1)

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

fullmodel = FullModel()

# Choose a loss function and an optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(fullmodel.parameters(), lr=0.001)

# Split the dataset: 80% for training, 20% for testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Training loop with evaluation on the test set
num_epochs = 5

for epoch in range(num_epochs):
    # Training phase
    fullmodel.train()
    running_loss = 0.0

    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
        # Zero the gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: compute model outputs
        outputs = fullmodel(batch_inputs)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Training Average Loss: {avg_train_loss:.4f}')

    # Testing phase
    fullmodel.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            outputs = fullmodel(batch_inputs)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            # For accuracy: get the index of the max log-probability
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] Test Average Loss: {avg_test_loss:.4f}, '
          f'Accuracy: {accuracy:.2%}\n')

print("Training complete!")