import numpy as np
import random, math

def sigmoid(x):
    y = 1/(1+math.exp(-x))
    # print(' The activation is ', y)
    return y

class Neuron:
    def __init__(self, numberOfWeights):
        self.weights = np.array([])
        for i in range(numberOfWeights):
            self.weights = np.append(self.weights, random.uniform(-1, 1))
        self.bias = random.uniform(-10, 10)
        
    def setActivation(self, activation, prevLayer):
        if activation != None:
            self.activation = activation
        elif prevLayer != None:
            self.activation = self.computeActivation(prevLayer)
        
        
    def setBias(self, bias):
        self.bias = bias
    
    def computeActivation(self, prevLayer):
        return sigmoid(np.dot(self.weights, prevLayer.activations) + self.bias)
    
    def computeZ(self, prevLayer):
        return (np.dot(self.weights, prevLayer.activations) + self.bias)
    
    def show(self):
        print('Weights: ',self.weights)
        print('Bias: ', self.bias)
        print('Activation: ', self.activation)
        
    def getActivation(self, prevLayer):
        return self.activation
    