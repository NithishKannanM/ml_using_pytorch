import torch 
import torch.nn as nn
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print(X.shape, y.shape)