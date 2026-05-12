import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),  # First layer: 10 inputs, 5 outputs
    nn.ReLU(),        # Activation function
    nn.Linear(5, 1)   # Second layer: 5 inputs, 1 output (linked to the first layer)
)

# Create a loss function and an optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss

optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer


# make training data
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randn(100, 1)   # 100 samples, 1 target value

# Train the model
for epoch in range(10): 
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
    
    # Print the loss for every epoch
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
