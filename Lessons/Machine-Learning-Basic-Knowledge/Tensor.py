import torch as torch;

print("=" * 50)
print("Tensors en PyTorch")
print("=" * 50)

#Tensor is a data structure that can hold data in multiple dimensions.


# Simple tensor
x = torch.tensor(5)
print("0D =", x)

# Tensor 1D vector
vector = torch.tensor([1, 2, 3, 4, 5])
print("1D =", vector)

# Tensor 2D matrice
matrice = torch.tensor([[1, 2, 3], 
                        [4, 5, 6]])
print("2D =", matrice)

# Shape of tensor
print("Shape of 2D matrice =", matrice.shape)
print("Value at [0, 1] =", matrice[0, 1])
print("\n")

# Basic operations
print("=" * 50)
print("Basic Operations")
print("=" * 50)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Addition
print("Addition =", a + b)

# Subtraction
print("Subtraction =", a - b)

# Multiplication
print("Multiplication =", a * b)

# Multipication with scalar
print("Multiplication with scalar =", a * 2)

# Complexe operation
print("Complexe operation =", torch.sqrt(torch.tensor([4.0, 9.0])))

# Mean
print("Mean = " + str(torch.mean(a)))

# Shape
print("Shape of a =", a.shape)