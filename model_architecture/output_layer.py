
import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.fc2 = nn.Linear(512, output_dim)  
        self.activation = nn.ReLU()
    
    def forward(self, x):
        
        if x.ndim != 2 or x.shape[1] != self.fc1.in_features:
            raise ValueError(f"Expected input tensor of shape (batch_size, {self.fc1.in_features}), got {x.shape}.")
        
        x = self.fc1(x)  
        x = self.activation(x)  
        x = self.fc2(x)  
        return x  



if __name__ == "__main__":

    # Simulated input from Feature Extraction Layer
    features = torch.rand(1,1600)
    
    input_dim = features.shape[1]  
    print(f"Input Dimension for Output Layer: {input_dim}")
    output_layer = OutputLayer(input_dim=input_dim, output_dim=2)  
    logits = output_layer(features)

    torch.save(output_layer.state_dict(), "output_layer.pth")
    print("Output Layer model state_dict saved as 'output_layer.pth'.")

    loaded_model = OutputLayer(input_dim=input_dim, output_dim=2)
    loaded_model.load_state_dict(torch.load("output_layer.pth"))
    print("Output Layer model reloaded successfully.")
    
    operator_probs = torch.softmax(logits, dim=1)
    print(f"Softmax probabilities: {operator_probs.tolist()}")

    # *Get predicted operator index*
    predicted_operator_idx = torch.argmax(operator_probs, dim=1)
    operators = ["unstack", "transpose"]  
    predicted_operator = [operators[i] for i in predicted_operator_idx.tolist()]

    print(f"Predicted Operator: {predicted_operator}")




