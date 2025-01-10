# import torch
# import torch.nn as nn
# import numpy as np

# # Input tensor: (n=101, m=50, d=423)
# input_tensor = torch.rand(101, 50, 423)  # Simulated output of the embedding layer

# # Define Dimension Reduction Layers
# class DimensionReductionLayer(nn.Module):
#     def __init__(self, in_channels=423, mid_channels=64, out_channels=32):
#         """
#         Dimension Reduction Layer reduces input dimensions using 1x1 convolutions.
#         Args:
#             in_channels: Initial input feature dimensions (default=423).
#             mid_channels: Intermediate dimension size (default=64).
#             out_channels: Final reduced dimension size (default=32).
#         """
#         super(DimensionReductionLayer, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
#         )
#         self.activation = nn.ReLU()  # Activation function
#         self.batch_norm1 = nn.BatchNorm2d(mid_channels)  # Batch normalization after first convolution
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)  # Batch normalization after second convolution

#     def forward(self, x):
#         # Apply the 1x1 convolutions with activation and normalization
#         x = self.conv1(x)  # Reduce dimensions from in_channels to mid_channels
#         x = self.batch_norm1(x)  # Batch normalization
#         x = self.activation(x)
        
#         x = self.conv2(x)  # Reduce dimensions from mid_channels to out_channels
#         x = self.batch_norm2(x)  # Batch normalization
#         x = self.activation(x)
#         return x

# # Validate input tensor shape
# assert input_tensor.shape == (101, 50, 423), "Input tensor must have shape (101, 50, 423)."

# # Reshape the input tensor for Conv2D
# # PyTorch Conv2D expects input in the shape (batch_size, channels, height, width)
# input_tensor_reshaped = input_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 423, 101, 50)

# # Apply the Dimension Reduction Layers
# dim_reduction_layer = DimensionReductionLayer()
# output_tensor = dim_reduction_layer(input_tensor_reshaped)  # Shape: (1, 32, 101, 50)

# # Reshape back to (n, m, d) for further processing
# output_tensor_final = output_tensor.squeeze(0).permute(1, 2, 0)  # Shape: (101, 50, 32)

# print("Output Tensor Shape after Dimension Reduction:", output_tensor_final.shape)

# # Detach from the computation graph and convert to NumPy
# output_tensor_np = output_tensor_final.detach().numpy()

# # Save the NumPy array to a file
# np.save("dimension_reduced_tensor.npy", output_tensor_np)
# print("Dimension-reduced tensor saved as 'dimension_reduced_tensor.npy'")

# # Load the saved NumPy file
# tensor_data = np.load("dimension_reduced_tensor.npy")

# # Print the shape of the loaded tensor
# print("Loaded Tensor Shape:", tensor_data.shape)

# # Save as CSV for easier inspection
# np.savetxt("dimension_reduced_tensor.csv", tensor_data.reshape(-1, tensor_data.shape[-1]), delimiter=",")
# print("Tensor saved as 'dimension_reduced_tensor.csv'")



import torch
import torch.nn as nn
import numpy as np


# Define Dimension Reduction Layer
class DimensionReductionLayer(nn.Module):
    def __init__(self, in_channels=423, mid_channels=64, out_channels=32):
        """
        Dimension Reduction Layer using two 1x1 convolution layers.
        Reduces input dimensions from in_channels → mid_channels → out_channels.
        """
        super(DimensionReductionLayer, self).__init__()
        # First convolution: 423 → 64
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        # Second convolution: 64 → 32
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, but got {x.ndim}D tensor. Shape: {x.shape}")
        # Apply first convolution, batch normalization, and activation
        x = self.activation(self.batch_norm1(self.conv1(x)))
        # Apply second convolution, batch normalization, and activation
        x = self.activation(self.batch_norm2(self.conv2(x)))
        return x


# Simulated Input Tensor (Embedding Layer Output)
if __name__ == "__main__":
    # Simulate input tensor of shape (n=101, m=50, d=423)
    input_tensor = torch.rand(49, 102, 423)

    # Reshape for Conv2D
    # Expected input for Conv2D: (batch_size, channels, height, width)
    n, m, d = input_tensor.shape  # (n, m, d)
    input_tensor_reshaped = input_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, d, n, m)

    # Initialize and Apply the Dimension Reduction Layer
    dim_reduction_layer = DimensionReductionLayer()
    output_tensor = dim_reduction_layer(input_tensor_reshaped)  # Output shape: (1, 32, n, m)

    # Reshape Back to Original Structure
    output_tensor_final = output_tensor.squeeze(0).permute(1, 2, 0)  # Shape: (n, m, out_channels)
    print("Output Tensor Shape after Dimension Reduction:", output_tensor_final.shape)

    # Detach the tensor from the computation graph and move to CPU
    output_tensor_final_cpu = output_tensor_final.detach().cpu()

    # Save the tensor in .pth format
    torch.save(output_tensor_final_cpu, "dimension_reduced_tensor.pth")
    print("Dimension-reduced tensor saved as 'dimension_reduced_tensor.pth'.")

    # Save the model state dictionary
    torch.save(dim_reduction_layer.state_dict(), "dimension_reduction_layer.pth")
    print("Dimension Reduction Layer state dictionary saved as 'dimension_reduction_layer.pth'.")

    # Optional: Load the saved .pth file to verify
    loaded_tensor = torch.load("dimension_reduced_tensor.pth")
    print("Loaded Tensor Shape:", loaded_tensor.shape)


# Save Tensor in .npy and .h5 Formats
# output_tensor_np = output_tensor_final.detach().cpu().numpy()
# np.save("dimension_reduced_tensor.npy", output_tensor_np)
# print("Dimension-reduced tensor saved as 'dimension_reduced_tensor.npy'.")

# with h5py.File("dimension_reduced_tensor.h5", "w") as hf:
#     hf.create_dataset("reduced_tensor", data=output_tensor_np)
# print("Tensor saved as 'dimension_reduced_tensor.h5'.")
