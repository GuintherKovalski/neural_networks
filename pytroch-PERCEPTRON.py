import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,1)
    def forward(self, x):
        x = self.fc1(x)
        return x
    
net = Net()
print(net)    
print(list(net.parameters()))
input = Variable(torch.randn([1,1,1]), requires_grad=True)
print(input)
out = net(input)

import torch.optim as optim
def criterion(out, label):
    return (label - out)**2
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

LOSS = []
for epoch in range(100):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):            
            LOSS.append(loss.data[0]) 
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))



training_inputs = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])

training_outputs = np.array([[1,0,1,0,1,0,1,0]]).T

training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])
                            

training_outputs = np.array([[0,1,1,0]]).T



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,1000)
        self.fc2 = nn.Linear(1000,1)
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x
    
net = Net()    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,2)
        self.fc3 = nn.Linear(2,1)
    def forward(self, x):
        x = np.cos(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return x
        
criterion = nn.MSELoss()
net = Net() 


for epoch in range(1000):
    for i in range(len(training_inputs)):
        X, Y = training_inputs[i],training_outputs[i]
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))    

print(net(Variable(torch.Tensor([[[1,1]]]))))
    
    
    
    
    
    


