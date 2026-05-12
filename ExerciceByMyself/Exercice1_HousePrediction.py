import torch
import torch.nn as nn
import torch.optim as optim

# Create synthetic data for house price prediction
x = torch.tensor([
    [65.0, 2.0],
    [72.0, 2.0],
    [85.0, 3.0],
    [90.0, 3.0],
    [95.0, 3.0],
    [110.0, 4.0],
    [120.0, 4.0],
    [130.0, 4.0],
    [145.0, 5.0],
    [160.0, 5.0],
    [175.0, 6.0],
    [190.0, 6.0],
    [210.0, 7.0],
    [230.0, 7.0],
    [250.0, 8.0],
    [70.0, 2.0],
    [88.0, 3.0],
    [105.0, 4.0],
    [125.0, 4.0],
    [155.0, 5.0],
    [170.0, 5.0],
    [185.0, 6.0],
    [205.0, 7.0],
    [225.0, 7.0],
    [240.0, 8.0],
    [260.0, 9.0],
    [75.0, 2.0],
    [98.0, 3.0],
    [115.0, 4.0],
    [135.0, 4.0]
], dtype=torch.float32)

# Output: house prices in dollars
y = torch.tensor([
    [120000.0],
    [128500.0],
    [145000.0],
    [158000.0],
    [165000.0],
    [195000.0],
    [215000.0],
    [230000.0],
    [275000.0],
    [310000.0],
    [355000.0],
    [390000.0],
    [470000.0],
    [520000.0],
    [610000.0],
    [125000.0],
    [152000.0],
    [185000.0],
    [225000.0],
    [295000.0],
    [340000.0],
    [385000.0],
    [455000.0],
    [510000.0],
    [590000.0],
    [690000.0],
    [132000.0],
    [172000.0],
    [205000.0],
    [245000.0]
], dtype=torch.float32)

# Saves stats
x_mean = x.mean(dim=0) # Calculate mean for each feature (size and bedrooms) to compare with new data
x_std = x.std(dim=0) # Calculate standard deviation for each feature (size and bedrooms) to compare with new data

# Normalize training data
x = (x - x_mean) / x_std # give the results of the normalization to x to be used in the training of the model


# Create a simple model
model = nn.Sequential(
    nn.Linear(2, 30),
    nn.ReLU(),
    nn.Linear(30, 16),
    nn.ReLU(),
     nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Create a loss function and an optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
# Train the model
for epoch in range(10000): 
    # Get predictions from the model
    predictions = model(x)  
    # Calculate the loss
    loss = loss_fn(predictions, y)  
    # bacckward pass : calculate gradients
    
    # Clear previous gradients
    optimizer.zero_grad()  
    
    # Backpropagation to calculate gradients
    loss.backward()   
         
    # Update model parameters based on gradients
    optimizer.step()       
    
    # Print the loss for every  10 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/10000, Loss: {loss.item():.4f}")
        
   
# Test the model with new data
new_data_1 = torch.tensor([[180.0, 3.0]])  # New house with 180 sqft and 3 bedrooms
new_data_1 = (new_data_1 - x_mean) / x_std

predicted_price = model(new_data_1)
print(f"Predicted price for a house with 180 sqft and 3 bedrooms: ${predicted_price.item():.2f}")

new_data_2 = torch.tensor([[200.0, 5.0]])  # New house with 200 sqft and 5 bedrooms
new_data_2 = (new_data_2 - x_mean) / x_std

predicted_price = model(new_data_2)

print(f"Predicted price for a house with 200 sqft and 5 bedrooms: ${predicted_price.item():.2f}")

new_data_3 = torch.tensor([[120.0, 2.0]])  # New house with 120 sqft and 2 bedrooms
new_data_3 = (new_data_3 - x_mean) / x_std

predicted_price = model(new_data_3)

print(f"Predicted price for a house with 120 sqft and 2 bedrooms: ${predicted_price.item():.2f}")

new_data_4 = torch.tensor([[80.0, 1.0]])  # New house with 80 sqft and 1 bedroom
new_data_4 = (new_data_4 - x_mean) / x_std

predicted_price = model(new_data_4)

print(f"Predicted price for a house with 80 sqft and 1 bedroom: ${predicted_price.item():.2f}")