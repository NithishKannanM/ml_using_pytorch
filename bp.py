import torch
x=torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0, requires_grad=True )

#forward pass
y_h = w*x
loss = (y_h - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)