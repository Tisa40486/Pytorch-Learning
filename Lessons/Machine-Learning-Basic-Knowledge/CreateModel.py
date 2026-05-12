import torch;
import torch.nn as nn;


class MyModel(nn.Module):
    
    def __init__(self):
       super().__init__()
       
       # First layer: takes 10 features and outputs 5
       self.fc1 = nn.Linear(10, 5)  
       
        # Second layer: takes 5 features and outputs 1
       self.fc2 = nn.Linear(5, 1) 
        
    def forward(self, x):
        x = self.fc1(x)  # Pass through first layer
        
        x = torch.relu(x)  # Apply ReLU activation
        
        x = self.fc2(x)  # Pass through second layer
        
        return x
    
# Create an instance of the model
model = MyModel()

# Create a random input tensor with 10 features
data = torch.randn(3, 10)  # Batch size of 3, 10 features

# Get predictions from the model
predictions = model(data)

print("=" * 50)
print("Predictions")
print("=" * 50)

print("Predictions Shape:", predictions.shape)

print("Predictions:", predictions)