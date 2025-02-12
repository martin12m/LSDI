
import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        
        super(OutputLayer, self).__init__()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(input_dim, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, output_dim)  # Second fully connected layer
        
        # Activation Function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        
        if x.ndim != 2 or x.shape[1] != self.fc1.in_features:
            raise ValueError(f"Expected input tensor of shape (batch_size, {self.fc1.in_features}), got {x.shape}.")
        
        x = self.fc1(x)  # First fully connected layer
        x = self.activation(x)  # Activation
        x = self.fc2(x)  # Second fully connected layer
        return x  # Raw logits; softmax can be applied externally



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




