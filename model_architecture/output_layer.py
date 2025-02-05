# import torch
# import torch.nn as nn

# class OutputLayer(nn.Module):
#     def __init__(self, input_dim=128 * (101 // 2), output_dim=270):
#         """
#         Output Layer for multi-task classification.
#         Args:
#             input_dim: Flattened input dimension from Feature Extraction Layer.
#             output_dim: Combined dimension for operator type and parameters (default=270).
#         """
#         super(OutputLayer, self).__init__()
        
#         # Fully Connected Layers
#         self.fc1 = nn.Linear(input_dim, 512)  # First fully connected layer
#         self.fc2 = nn.Linear(512, output_dim)  # Second fully connected layer
        
#         # Activation Function
#         self.activation = nn.ReLU()
        
#         # Softmax for output probabilities
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x):
#         """
#         Forward pass for the output layer.
#         Args:
#             x: Input tensor of shape (batch_size, input_dim).
#         Returns:
#             Normalized probability tensor of shape (batch_size, output_dim).
#         """
#         x = self.fc1(x)  # First fully connected layer
#         x = self.activation(x)  # Activation
#         x = self.fc2(x)  # Second fully connected layer
#         x = self.softmax(x)  # Normalize predictions
#         return x


# # Example Usage
# if __name__ == "__main__":
#     # Simulated input from Feature Extraction Layer
#     feature_tensor = torch.rand(32, 128 * (101 // 2))  # Batch of 32, feature vectors

#     # Initialize the Output Layer
#     output_layer = OutputLayer(input_dim=128 * (101 // 2), output_dim=270)

#     # Forward pass
#     output = output_layer(feature_tensor)  # Shape: (32, 270)

#     print("Output Shape:", output.shape)  # Expected: (batch_size, output_dim)

#     # Save the model
#     torch.save(output_layer.state_dict(), "output_layer.pth")
#     print("Output Layer model saved as 'output_layer.pth'")

#     # Simulated input from Feature Extraction Layer
#     features = torch.rand(32, 128 * (101 // 2))  # Output from Feature Extraction Layer

#     # Initialize and forward pass through the Output Layer
#     output_layer = OutputLayer(input_dim=128 * (101 // 2), output_dim=270)
#     output = output_layer(features)

#     print("Final Output Shape:", output.shape)  # Expected: (32, 270)



import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=270):
        """
        Output Layer for multi-task classification, producing both operator-type and parameter predictions.
        
        Args:
            input_dim: Flattened input dimension from Feature Extraction Layer (6528 for your case).
            output_dim: Combined dimension for operator type (8 classes) and operator parameters (262 parameters).
                        Default = 270.
        """
        super(OutputLayer, self).__init__()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(input_dim, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, output_dim)  # Second fully connected layer
        
        # Activation Function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass for the output layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            Raw logits tensor of shape (batch_size, output_dim).
        """
        if x.ndim != 2 or x.shape[1] != self.fc1.in_features:
            raise ValueError(f"Expected input tensor of shape (batch_size, {self.fc1.in_features}), got {x.shape}.")
        
        x = self.fc1(x)  # First fully connected layer
        x = self.activation(x)  # Activation
        x = self.fc2(x)  # Second fully connected layer
        return x  # Raw logits; softmax can be applied externally


# Example Usage
if __name__ == "__main__":

    # *Simulated input from Feature Extraction Layer*
    features = torch.rand(1,1600)
    # *Calculate input dimension for OutputLayer*
    input_dim = features.shape[1]  
    print(f"Input Dimension for Output Layer: {input_dim}")

    # *Initialize the Output Layer for 2 classes*
    output_layer = OutputLayer(input_dim=input_dim, output_dim=2)  # Only 2 operators

    # *Pass extracted features through Output Layer*
    logits = output_layer(features)

    # Save the model state_dict
    torch.save(output_layer.state_dict(), "output_layer.pth")
    print("Output Layer model state_dict saved as 'output_layer.pth'.")

    # Reload the model
    loaded_model = OutputLayer(input_dim=input_dim, output_dim=2)
    loaded_model.load_state_dict(torch.load("output_layer.pth"))
    print("Output Layer model reloaded successfully.")
    
    # *Convert logits to probabilities using softmax*
    operator_probs = torch.softmax(logits, dim=1)
    print(f"Softmax probabilities: {operator_probs.tolist()}")

    # *Get predicted operator index*
    predicted_operator_idx = torch.argmax(operator_probs, dim=1)

    # *Define the two operator classes*
    operators = ["unstack", "transpose"]  # Only 2 classes
    predicted_operator = [operators[i] for i in predicted_operator_idx.tolist()]

    print(f"Predicted Operator: {predicted_operator}")




