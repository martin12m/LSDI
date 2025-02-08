import torch
import torch.nn as nn
import numpy as np


class DimensionReductionLayer(nn.Module):
    def __init__(self, in_channels=423, mid_channels=64, out_channels=32):
        super(DimensionReductionLayer, self).__init__()
        
        # First convolution: 423 → 64
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        
        # Second convolution: 64 → 32
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor (batch_size, channels, height, width), but got {x.ndim}D tensor. Shape: {x.shape}")
        
        x = self.activation(self.batch_norm1(self.conv1(x)))
        print("Output after first convolution:", x.shape)
        
        x = self.activation(self.batch_norm2(self.conv2(x)))
        print("Output after second convolution:", x.shape)
        
        return x
    

if __name__ == "__main__":
    # Input Feature Tensor Shape from Embedding Layer
    final_features = torch.rand(101, 50, 423)  

    # Reshape input tensor to (batch_size, channels, height, width)
    input_tensor = final_features.unsqueeze(0)  
    input_tensor = input_tensor.permute(0, 3, 1, 2) 
    print(f"Input Tensor Shape: {input_tensor.shape}")

    # Initialize dimension reduction layer
    in_channels = input_tensor.shape[1]  # 423
    dim_reduction_layer = DimensionReductionLayer(in_channels=in_channels, mid_channels=64, out_channels=32)


    output_tensor = dim_reduction_layer(input_tensor)
    print(f"Output Tensor Shape: {output_tensor.shape}")
    output_tensor_final_cpu = output_tensor.detach().cpu()

    torch.save(output_tensor_final_cpu, "dimension_reduced_tensor.pth")
    print("Dimension-reduced tensor saved as 'dimension_reduced_tensor.pth'.")

    torch.save(dim_reduction_layer.state_dict(), "dimension_reduction_layer.pth")
    print("Dimension Reduction Layer state dictionary saved as 'dimension_reduction_layer.pth'.")

    loaded_tensor = torch.load("dimension_reduced_tensor.pth")
    print("Loaded Tensor Shape:", loaded_tensor.shape)