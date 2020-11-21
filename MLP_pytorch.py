import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = np.linspace(-10, 10, num=60)
y = np.cos(x)
plt.plot(y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,1)
    def forward(self, x):
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 1)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x  
    
net = Net()
net = MLP()
input = Variable(torch.randn([1,1,1]), requires_grad=True)
out = net(input)

def criterion(out, label):
    return (label - out)**2


optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.8)
optimizer = optim.Adam(net.parameters())
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]
y_hat = []
x_hat = []
LOSS = []
for epoch in range(10000):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):            
            LOSS.append(loss.data[0]) 
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))           
y_hat = []
y_true = []
for x,y in data:
    X, Y = Variable(torch.FloatTensor([x]), requires_grad=True), Variable(torch.FloatTensor([y]), requires_grad=False)
    X = Variable(torch.FloatTensor([x]), requires_grad=True)
    y_hat.append(net(X))  
    y_true.append(Y)
plt.plot(y_hat)
plt.plot(y_true)


net = MLP()
input = Variable(torch.randn([1,1,1]), requires_grad=True)
out = net(input)

x = np.linspace(-10, 10, num=600)
y = np.cos(x)
for epoch in range(1000):
    for i in range(len(x)):
        X, Y = x[i],y[i]
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):            
            LOSS.append(loss.data[0]) 
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))       
y_hat = []
y_true = []
for i in range(len(x)):
    X, Y = Variable(torch.FloatTensor([x[i]]), requires_grad=True), Variable(torch.FloatTensor([y[i]]), requires_grad=False)
    X = Variable(torch.FloatTensor([x[i]]), requires_grad=True)
    y_hat.append(net(X))  
    y_true.append(Y)
plt.plot(y_hat)
plt.plot(y_true)


###############################################################################

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(60, 600),
            nn.Sigmoid(),
            nn.Linear(600, 60)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x  
    
net = MLP()
input = Variable(torch.randn([1,1,1]), requires_grad=True)
out = net(input)

x = np.linspace(-10, 10, num=60)
y = np.cos(x)
for epoch in range(1000):
    X, Y = x,y
    X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    if (i % 10 == 0):            
        LOSS.append(loss.data[0]) 
        print("Epoch {} - loss: {}".format(epoch, loss.data[0]))       

y_hat = []
y_true = []
for i in range(len(x)):
    X, Y = Variable(torch.FloatTensor([x[i]]), requires_grad=True), Variable(torch.FloatTensor([y[i]]), requires_grad=False)
    X = Variable(torch.FloatTensor([x[i]]), requires_grad=True)
    y_hat.append(net(X))  
    y_true.append(Y)
plt.plot(y_hat)
plt.plot(y_true)































