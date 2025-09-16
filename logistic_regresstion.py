import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#load data
df = datasets.load_breast_cancer()

X,y = df.data, df.target

n_samples, n_features = X.shape

x_train, y_train ,x_test, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

#model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features) 

#loss and optimizer
loss = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    y_predicted = model(x_train)
    l = loss(y_predicted, y_train)
    
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}: loss = {l.item():.4f}')
#evaluation
with torch.no_grad():
    model.eval()
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
