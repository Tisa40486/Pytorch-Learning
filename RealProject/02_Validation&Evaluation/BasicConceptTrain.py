import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

file = pd.read_csv('./better_data.csv')
data = pd.DataFrame(file)


X = data[['Age', 'Experience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, # target & feature
    test_size=0.2,
    shuffle=True
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: ${mae:.2f}") 

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

rmse = np.sqrt(mse)
print(f"RMSE: ${rmse:.2f}") 

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

scores = cross_val_score(
    model,               
    X, y,                   # target and feature 
    cv=20,                   # 5 splits
    scoring='r2'           # Use R2 metric
)

print(f"Fold scores: {scores}")
print(f"Mean score: {scores.mean():.4f}")
print(f"Std dev: {scores.std():.4f}")

train_sizes, train_scores, val_scores = learning_curve(
    model, 
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.plot(train_sizes, val_mean, label='Validation score', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curve - Detect Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()