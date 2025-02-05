import torch
import torch.nn as nn
import numpy as np


# Define Dimension Reduction Layer

class DimensionReductionLayer(nn.Module):
    def __init__(self, in_channels=423, mid_channels=64, out_channels=32, num_oprators=4, sequence_length=None):
        
        super(DimensionReductionLayer, self).__init__()
        
        # First convolution: 423 → 64
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
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
        print("Output after first convolution:", x.shape)
        
        # Apply second convolution, batch normalization, and activation
        x = self.activation(self.batch_norm2(self.conv2(x)))
        print("Output after second convolution:", x.shape)
        
        return x
    
    
    

# Simulated Input Tensor (Embedding Layer Output)
if __name__ == "__main__":
    
    #input Feature Tensor Shape from Embedding Layer
    final_features = torch.rand(101, 50, 423)


    input_tensor = torch.rand(final_features.shape)
    n, m, d = input_tensor.shape  # (n, m, d)
    input_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0)
    input_tensor_reshaped = input_tensor.permute(0, 3, 1, 2)
    print(f"Input Tensor Shape: {input_tensor_reshaped.shape}")
    batch_size, in_channels, height, width = input_tensor_reshaped.shape
    dim_reduction_layer = DimensionReductionLayer(in_channels=in_channels, mid_channels=64, out_channels=32)
    output_tensor = dim_reduction_layer(input_tensor_reshaped)
    print(f"Output Tensor Shape: {output_tensor.shape}")
    
    
    

    # Detach the tensor from the computation graph and move to CPU
    output_tensor_final_cpu = output_tensor.detach().cpu()

    # Save the tensor in .pth format
    torch.save(output_tensor_final_cpu, "dimension_reduced_tensor.pth")
    print("Dimension-reduced tensor saved as 'dimension_reduced_tensor.pth'.")

    # Save the model state dictionary
    torch.save(dim_reduction_layer.state_dict(), "dimension_reduction_layer.pth")
    print("Dimension Reduction Layer state dictionary saved as 'dimension_reduction_layer.pth'.")

    # Optional: Load the saved .pth file to verify
    loaded_tensor = torch.load("dimension_reduced_tensor.pth")
    print("Loaded Tensor Shape:", loaded_tensor.shape)
