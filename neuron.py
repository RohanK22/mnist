import numpy as np
import random

class Neuron:
    def __init__(self, numberOfWeights):
        self.weights = np.array([])
        for i in range(numberOfWeights):
            self.weights = np.append(self.weights, random.uniform(-1, 1))
        self.bias = random.uniform(-10, 10)
        self.setActivation(0)
        
    def setActivation(self, activation):
        self.actication = activation
    
    def show(self):
        print('Weights: ',self.weights)
        print('Bias: ', self.bias)
        print('Activation: ', self.activation)