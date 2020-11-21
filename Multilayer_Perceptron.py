import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
    #return x * (1 - x)
    
def sigmoid_sec_derivative(x):
    #f'(h(x))*h'(x)
    return sigmoid_derivative(a)*sigmoid_derivative(x)

def linear(x):
    for i in range(len(x)):
        if x[i]<0:
            x[i] = 0
    return x
def lin_deriv(x):
    for i in range(len(x)):
        if x[i]<=0:
            x[i] = 0
        else:
            x[i]=1
    return x
            
def lin_sig_derivate(x): 
    return sigmoid_derivative(linear(x))*lin_deriv(x)
    


# input dataset
training_inputs = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])[::-1]
    
s1 = training_inputs.shape[1]
s2 = training_inputs.shape[0]  
# output dataset
training_outputs = np.array([[1,0,0,1,0,1,0,1]]).T

# seed random numbers to make calculation
np.random.seed(1)
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights1 = 10 * np.random.random((s1,1)) - 5

np.random.seed(2)
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights2 = 20 * np.random.random((s1,1)) - 10

np.random.seed(3)
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights3 = 100 * np.random.random((s1,1)) - 5

np.random.seed(4)
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights4 = 2 * np.random.random((s1,1)) - 1

np.random.seed(4)
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights_out = 2 * np.random.random((4,1)) - 1

#print('Random starting synaptic weights: ')
#print(synaptic_weights)
acc = []
loss = []
# Iterate 10,000 times
for iteration in range(10000):
    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs1 = linear(np.dot(input_layer, synaptic_weights1)).astype(float)
    
    outputs2 = linear(np.dot(input_layer, synaptic_weights2)).astype(float)
    
    outputs3 = linear(np.dot(input_layer, synaptic_weights3)).astype(float)
    
    outputs4 = linear(np.dot(input_layer, synaptic_weights4)).astype(float)
    
    out_l1 = np.array([outputs1,outputs2,outputs3,outputs4]).reshape(4,s2).T
    
    outputs = sigmoid(np.dot(out_l1, synaptic_weights_out)).astype(float)
    
    #synaptic_weights_out

    # how much did we miss?
    error = training_outputs - sigmoid(np.dot(out_l1, synaptic_weights_out)).astype(float)

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments1 = error * lin_deriv(outputs1)
    
    adjustments2 = error * lin_deriv(outputs2)
    
    adjustments3 = error * lin_deriv(outputs3)
    
    adjustments4 = error * lin_sig_derivate(outputs)

    # update weights
    synaptic_weights1 += np.dot(input_layer.T, 0.0001*adjustments1)
    synaptic_weights2 += np.dot(input_layer.T, 0.0001*adjustments2)
    synaptic_weights3 += np.dot(input_layer.T, 0.0001*adjustments3)
    synaptic_weights4 += np.dot(input_layer.T, 0.0001*adjustments4)
    synaptic_weights_out += np.dot(out_l1.T,0.0001*adjustments4)
    acc.append(sum((outputs>0.5)*1 == training_outputs)/len(training_outputs)) 
    loss.append(abs(error).mean()*5)


print('Synaptic weights after training: ')
print(synaptic_weights1)
print(synaptic_weights2)
print(synaptic_weights3)
print(synaptic_weights_out)

print("Output After Training:")
print((sigmoid(np.dot(out_l1, synaptic_weights_out)).astype(float)>0.499)*1)
print("True class:")
print(training_outputs)
sigmoid(np.dot(out_l1, synaptic_weights_out))

import matplotlib.pyplot as plt
plt.plot(loss)
plt.plot(acc)


#grit
#pcgritti