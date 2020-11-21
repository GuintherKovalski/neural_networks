import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - tanh(x)**2

np.random.seed(1)
x = np.linspace(0, 10, 200)[:, None]       # [batch, 1]
y = np.cos(x)*x + np.random.rand(200).reshape(200,1)                                # [batch, 1]

learning_rate = 0.0001
NL1 = 6
NL2 = 4

w1 = np.random.uniform(0, 1, (1, NL1 ))
w2 = np.random.uniform(0, 1, (NL1 , NL2))
w3 = np.random.uniform(0, 1, (NL2, 1))
b1 = np.full((1, NL1 ), 0.1)
b2 = np.full((1, NL2), 0.1)
b3 = np.full((1, 1), 0.1)
LOSS = []


for i in range(20000):
    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = tanh(z2)
    z3 = a2.dot(w2) + b2
    a3 = tanh(z3)
    z4 = a3.dot(w3) + b3

    cost = np.sum((z4 - y))
    #if cost == a

    # backpropagation
    z4_delta = z4 - y
    dw3 = a3.T.dot(z4_delta)
    db3 = np.sum(z4_delta, axis=0, keepdims=True)

    z3_delta = z4_delta.dot(w3.T) * derivative_tanh(z3)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta, axis=0, keepdims=True)

    z2_delta = z3_delta.dot(w2.T) * derivative_tanh(z2)
    dw1 = x.T.dot(z2_delta)
    db1 = np.sum(z2_delta, axis=0, keepdims=True)

    # update parameters
    for param, gradient in zip([w1, w2, w3, b1, b2, b3], [dw1, dw2, dw3, db1, db2, db3]):
        param -= learning_rate * gradient

    print(cost)
    
    LOSS.append(sum((((z4 - y)/len(y))**2)**0.5))
    
    if i%1000==0:
        plt.plot(x, z4)
        plt.plot(x,y)
        plt.savefig('img/MLP/MLP'+str(i)+'.jpg')
        plt.clf()

plt.plot(x, z4)
plt.plot(x,y)
plt.show()
plt.plot(LOSS)

#plt.plot(sum((a1.dot(w1) + b1).T))
#plt.plot(sum((a2.dot(w2) + b2).T))
#plt.plot(sum((tanh(z3)).T))
#plt.plot(sum((a3.dot(w3) + b3).T))
#for param, gradient in zip([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]):
#    print(param)
#    print(gradient)

'''
def cost(parameters, *args):
    w1,w2,w3,b1,b2,b3 = parameters
    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = tanh(z2)
    z3 = a2.dot(w2) + b2
    a3 = tanh(z3)
    z4 = a3.dot(w3) + b3
    result = np.sum((z4 - y))
    return result


def rmse(x,w1,w2,w3,b1,b2,b3):
    
    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = tanh(z2)
    z3 = a2.dot(w2) + b2
    a3 = tanh(z3)
    z4 = a3.dot(w3) + b3
    cost = np.sum((z4 - y))
    
    return cost

'''














