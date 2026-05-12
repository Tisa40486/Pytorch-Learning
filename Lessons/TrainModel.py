import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 8),  # First layer: 10 inputs, 8 outputs
    nn.ReLU(),         # Activation function
    nn.Linear(8, 5),   # Second layer: 8 inputs, 5 outputs
    nn.ReLU(),         # Activation function
    nn.Linear(5, 5),   # Third layer: 5 inputs, 5 outputs
    nn.ReLU(),         # Activation function
    nn.Linear(5, 3),   # Fourth layer: 5 inputs, 3 outputs
    nn.ReLU(),         # Activation function
    nn.Linear(3, 1)    # Fifth layer: 3 inputs, 1 output (linked to the fourth layer)
)

# Create a loss function and an optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss

optimizer = optim.SGD(model.parameters(), lr=0.0001)  # Stochastic Gradient Descent optimizer


# make training data
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randn(100, 1)   # 100 samples, 1 target value

# Train the model
for epoch in range(1000): 
    # Get predictions from the model
    predictions = model(X_train)  
    # Calculate the loss
    loss = loss_fn(predictions, y_train)  
    # bacckward pass : calculate gradients
    
    # Clear previous gradients
    optimizer.zero_grad()  
    
    # Backpropagation to calculate gradients
    loss.backward()   
         
    # Update model parameters based on gradients
    optimizer.step()       
    
    # Print the loss for every  10 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")
