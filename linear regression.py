import torch
import numpy as np

#f = w*x
#f = 2*x
X = torch.tensor([1.0,2.0,3.0],dtype=torch.float32)
Y = torch.tensor([2.0,4.0,6.0],dtype=torch.float32)
w = 0.0
#model prediction
def forward(x):
    return w*x

#loss = MSE
def loss(y,y_pred):
    return ((y - y_pred) ** 2).mean()              #j = 1/n*(y-y_pred)**2 

#gradient 
def gradient(x,y,y_pred):                       
    return torch.dot(2*x, y_pred - y).mean()       #dj/dw = 1/n*2x*(y-y_pred)


print(f'Prediction before training: f(500283) = {forward(500283):.3f}')

#training
learning_rate = 0.01
n_iters = 100

for epoch  in range(n_iters):
    #forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y,y_pred)

    #gradient
    grad = gradient(X,Y,y_pred)
    w -= learning_rate * grad 

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.5f}')

print(f'Prediction after training: f(500283) = {forward(500283):.3f}')