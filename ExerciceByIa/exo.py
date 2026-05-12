import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===== DONNÉES =====
# x = surface, y = prix (relation : y = 50*x + 20)
x = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.4], [4.0], [5.0], [5.2],
    [5.5], [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0]
])
y = torch.tensor([
    [70.0], [95.0], [120.0], [145.0], [170.0], [195.0], [220.0], [245.0], [270.0],
    [295.0], [320.0], [345.0], [370.0], [395.0], [420.0], [445.0], [470.0], [495.0], [520.0]
])
# ===== MODÈLE =====
model = nn.Linear(1, 1)  # y = w*x + b

# ===== ENTRAÎNEMENT =====
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    pred = model(x)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.2f}")

# ===== RÉSULTATS =====
print(f"\nPoids (slope): {model.weight.item():.2f}")
print(f"Biais: {model.bias.item():.2f}")

with torch.no_grad():
    pred = model(x)
    for i in range(len(x)):
        print(f"x={x[i].item():.1f}, prédiction={pred[i].item():.1f}, vrai={y[i].item():.1f}")

# ===== PLOT =====
plt.scatter(x.numpy(), y.numpy(), label='Datas' , color='violet')
plt.plot(x.numpy(), pred.numpy(), color='red', label='Prédiction')
plt.legend()
plt.xlabel('Surface')
plt.ylabel('Price')
plt.title('Simple linear regression')
plt.savefig('simple_result.png')
print("\n✓ Graphique sauvegardé : simple_result.png")