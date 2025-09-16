import torch.nn as nn
import torch


#f = w*x
#f = 2*x
X = torch.tensor([[1.0],[2.0],[3.0]],dtype=torch.float32)
Y = torch.tensor([[2.0],[4.0],[6.0]],dtype=torch.float32)

X_test = torch.tensor([[4.0]],dtype=torch.float32)

input_size, output_size = X.shape[1], Y.shape[1]

model = nn.Linear(input_size, output_size)

#training
learning_rate = 0.01
n_iters = 2500

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch  in range(n_iters):   
    #forward pass
    y_pred = model(X)

    #loss
    l = loss(Y,y_pred)

    #backward pass
    l.backward()

    #update gradient
    optimizer.step()

    #to update the gradient to zero
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w.item():.3f}, b = {b.item():.3f}, loss = {l.item():.5f}')

print(f'Prediction after training: f({X_test.item():.3f}) = {model(X_test).item(): .3f}')