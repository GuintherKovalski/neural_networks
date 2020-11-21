 

# sigmoid deimport numpy as np
import matplotlib.pyplot as plt
import time


x = np.linspace(-10,10,100)
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
    
# input dataset
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

# output dataset
training_outputs = np.array([[0,0,0,1]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 100 * np.random.random((2,1)) - 50

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
loss = []
acc = []
bias = 1
for iteration in range(10000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    z = sigmoid(np.dot(input_layer, synaptic_weights[:input_layer.shape[1]])+bias)

    # how much did we miss?
    #error = training_outputs - outputs
    error = 2*(training_outputs - z)*(-1)
    loss.append(abs(error).mean()*5)
    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(z)*0.1

    # update weights
    synaptic_weights = synaptic_weights - np.dot(input_layer.T, adjustments)
    bias = bias - np.mean(adjustments)
    
    acc.append(sum((z>0.5)*1 == training_outputs)/len(training_outputs)) 
    
    x_out,y_out = sum((input_layer*synaptic_weights.T+bias).T),sigmoid( sum((input_layer*synaptic_weights.T+bias).T))
    
                                                                  
    if iteration%100 == 0:
       #plt.scatter(sum((input_layer*synaptic_weights.T).T),sigmoid( sum((input_layer*synaptic_weights.T).T) ),c='r' )
       plt.scatter(x_out[0],y_out[0],c='r' )
       plt.scatter(x_out[1],y_out[1],c='g' )
       plt.scatter(x_out[2],y_out[2],c='b' )
       plt.scatter(x_out[3],y_out[3],c='y' )
       plt.plot(x,sigmoid(x))
       plt.legend(['sigmoid','0,0','0,1','1,0','1,1'])
       plt.savefig('trainXOR/'+str(time.time())+'.jpg')
       
       plt.show()
       plt.cla()

plt.plot(loss)
plt.plot(acc)
plt.title('mean loss')
plt.xlabel('epochs')
plt.legend(['loss','accuracy'])
plt.savefig('trainXOR/train.jpg')

print('Synaptic weights after training: ')
print(synaptic_weights)
print(bias)

print("Output After Training:")
print(z>0.4)

##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
x = np.linspace(-4,4,2000)
y = x**2

dydx = (y[1:]-y[:-1])/(8/2000)
for i in range(1,2000,50):
    a = round(dydx[i],2)
    b = round((y[i]-dydx[i]*x[i]),2)
    
    plt.figure(figsize=(10,6))
    plt.grid()
    plt.plot(y)
    plt.plot(dydx[i]*x+(y[i]-dydx[i]*x[i]))
    plt.ylim(0,16)
    plt.legend(['loss','gradient'])
    plt.title('loss function')
    if b<0:
        plt.text(600, 10,'y='+str(a)+'x'+str(b), fontsize=30)
    if b>0:
        plt.text(600, 10,'y='+str(a)+'x+'+str(b), fontsize=14)
    plt.xlabel('w1',fontsize=30)
    plt.ylabel('loss',fontsize=30)
    plt.savefig('trainXOR/'+str(time.time())+'.jpg')
    plt.show()


for i in list(range(1,2000,50))[::-1]:
    a = round(dydx[i],2)
    b = round((y[i]-dydx[i]*x[i]),2)
    
    plt.figure(figsize=(10,6))
    plt.grid()
    plt.plot(y)
    plt.plot(dydx[i]*x+(y[i]-dydx[i]*x[i]))
    plt.ylim(0,16)
    plt.legend(['loss','gradient'])
    plt.title('loss function')
    if b<0:
        plt.text(600, 10,'y='+str(a)+'x'+str(b), fontsize=30)
    if b>0:
        plt.text(600, 10,'y='+str(a)+'x+'+str(b), fontsize=14)
    plt.xlabel('w1',fontsize=30)
    plt.ylabel('loss',fontsize=30)
    plt.savefig('trainXOR/'+str(time.time())+'.jpg')
    plt.show() 
    




##############################################################################


training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])
# output dataset
training_outputs = np.array([[0,0,0,1]]).T

lista = np.linspace(-25,25,300).tolist()
loss_landscape = []
for wi in lista:
    bias = wi
    outputs = sigmoid(np.dot(input_layer, synaptic_weights[:2])+bias)
    loss_landscape.append(sum(abs(training_outputs-outputs)))

plt.plot(loss_landscape)

###############################################################################
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,100)

import numpy as np
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
   
def normal(x,mu,sigma,curtose):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )*curtose

def normal_derivative(x,mu,sigma,curtose):
    return (( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )) * (2**0.5)*np.pi**(-1/4)*x*curtose

def agg_der(x,mu,sigma,curtose):
    y1 = normal_derivative(x,mu,sigma,curtose)
    y2 = sigmoid_derivative(x) 
    return y2-y1
    
def agg_act(x,mu,sigma,curtose):
    y1 = normal(x,mu,sigma,curtose)
    y2 = sigmoid(x) 
    return y2-y1

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
    
# input dataset
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

# output dataset
training_outputs = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 100 * np.random.random((2,1)) - 50

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
loss = []
acc = []


for iteration in range(2000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = agg_act( np.dot(input_layer, synaptic_weights[:input_layer.shape[1]])  , 4,2, 4)

    # how much did we miss?
    error = training_outputs - outputs
    loss.append(abs(error).mean()*5)
    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * agg_der(outputs,2,2,4)*0.1

    # update weights
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)
    
    acc.append(sum((outputs>0.5)*1 == training_outputs)/len(training_outputs)) 
    
    if iteration%100 == 0:
       plt.scatter( sum((input_layer*synaptic_weights.T).T) , agg_act( sum((input_layer*synaptic_weights.T).T)  ,4,2,4) ,c='r' )
       plt.plot(x,agg_act(x,4,2,4))
       plt.show()
       plt.cla()

    
    

plt.plot(loss)
plt.plot(acc)
plt.title('mean loss')
plt.xlabel('epochs')
plt.legend(['loss','accuracy'])
print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs>0.4)


z = sum((training_inputs*synaptic_weights.T).T)


normal(z,mu,sigma,1)<0.2





##w = w - del(E)/del(W)



##################################################################################################
##################################################################################################


training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

# output dataset
training_outputs = np.array([[1,1,0,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 1 * np.random.random((2,1)) - 0.5

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
loss = []
acc = []

mu = 0
sigma = 0.5


input_layer = training_inputs



def de_dy(y,y_hat):
    return -2(y-y_hat)

def de_dz(de_dy, input_sum):
    return de_dy * sigmoid_derivate(input_sum)


for iteration in range(20000):
    outputs = sigmoid( sum((input_layer*synaptic_weights.T).T) )
    error = training_outputs.T - outputs.T
    
    error = (training_outputs.T - outputs.T)**2
    
    
    
    loss.append(abs(error).mean())
    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    
    adjustments = (error * sigmoid_derivative(outputs)*0.1)[0]
    
    #adjustments = error * agg_der(outputs,mu,sigma,1)*0.1
    #adjustments = error * normal_derivative(outputs,mu,sigma,1)*0.1
    
    #normal
    #adjustments = error * normal_derivative(outputs,mu,sigma,1)*0.1
    
    mu = mu + adjustments.mean()*0.01
    sigma = sigma + adjustments.mean()*0.1
    
    
    #adjustments = error * agg_der(x,0,0.5,1) * 0.1

    # update weights
    for backprop in adjustments:
        synaptic_weights = synaptic_weights +  backprop   #+ np.dot(input_layer.T, adjustments)
    
    acc.append( ( (outputs>0.2)*1 == training_outputs.T).sum() / len(training_outputs)   ) 
    #MU.append(synaptic_weights[0])
    #SIGMA.append(synaptic_weights[1])
    
    if iteration%100 == 0:
        plt.scatter(sum((input_layer*synaptic_weights.T).T),sigmoid( sum((input_layer*synaptic_weights.T).T) ),c='r' )
        plt.plot(x,sigmoid(x))
        plt.show()
        plt.cla()
    
    

plt.plot(loss)
plt.plot(acc)
plt.title('mean loss')
plt.xlabel('epochs')
plt.legend(['loss','accuracy'])
print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs>0.4)


z = sum((training_inputs*synaptic_weights.T).T)


normal(z,mu,sigma,1)<0.2
































'''

# Define input layer
input_layer = np.array([1,1])
# Normalize the product of the input layer with the synaptic weights
outputs = sigmoid(np.dot(input_layer, synaptic_weights ))
print(outputs>0.4)

















###################################################################################



import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
    #return x * (1 - x)

#sigmoide = 1/(1+e^-x)     f = g(x)/h(x) .: (h(x)g'(x) - g(x)'h(x))/(h(x)²)  
#(e^-x)/(1+e^-x)^2

# input dataset
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

# output dataset
training_outputs = np.array([[0,1,1,1]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 100 * np.random.random((5,1)) - 50

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
loss = []
acc = []
for iteration in range(6000):

    # Feed Forward
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    #outputs = sigmoid(np.dot(input_layer, synaptic_weights[:input_layer.shape[1]]))

    outputs = sigmoid(sum((input_layer[0,:]*synaptic_weights[:]).T))
    # how much did we miss?
    error = training_outputs - outputs.reshape(4,1)
    loss.append(abs(error).mean())
    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error.T * (sigmoid_derivative(outputs))

    # update weights
    synaptic_weights +=  adjustments.T
    
    acc.append(sum(sum((outputs>0.9)*1 == training_outputs.T))/len(training_outputs)) 


import matplotlib.pyplot as plt
plt.plot(loss)
plt.plot(acc)
plt.title('mean loss')
plt.xlabel('epochs')
plt.legend(['loss','accuracy'])
print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs>0.4)

'''




































