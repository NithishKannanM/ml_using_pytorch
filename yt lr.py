import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = x.shape

#model
model = nn.Linear(n_features,1)

#loss and optimizer
loss = nn.MSELoss()
learning_rate = 0.01
n_iters = 1000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(n_iters):
    #forward pass and loss
    model.train()
    y_predicted = model(x)
    l = loss(y, y_predicted)

    #backward pass
    l.backward()

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}: loss = {l.item():.4f}')

#plot
predicted = model(x).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()