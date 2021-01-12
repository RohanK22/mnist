import pandas as pd
import numpy as np
import random

trainingExamples = pd.read_csv ('./data/mnist_train.csv')
print('Loaded training examples')

# 1D Array of all the labels to the img data 
ys = trainingExamples.iloc[:, 0].to_numpy()

# 2D Array of all the images (28 x 28) with their grayscale values
trainingSet = trainingExamples.iloc[:, 1:].to_numpy()


class Network:
    # layers
    def __init__(self):
        self.layer1 = Layer(784, None)
    
    def train(self):
        print('trining')
    def predict(self):
        print('Output is ')
        
class Layer:
    def __init__(self, numberOfNeurons, prevLayer):
        # Number of neurons in a layer
        self.numberOfNeurons = numberOfNeurons
        # Array of neuron objects
        self.neurons = []
        if prevLayer == None:
            numberOfWeights = 0
        else:
            numberOfWeights = prevLayer.numberOfNeurons
        for i in range(numberOfNeurons):
            newNeuron = Neuron(numberOfWeights)
            self.neurons.append(newNeuron)
        
        # Activation Array
        # Weights Array
        # Biases Array
            
    def show(self):
        print('Number of Neurons: ', self.numberOfNeurons)
        print('Neuron Objects: ', self.neurons)

class Neuron:
    def __init__(self, numberOfWeights):
        self.weights = np.array([])
        for i in range(numberOfWeights):
            self.weights = np.append(self.weights, random.uniform(-1, 1))
        self.bias = random.uniform(-10, 10)
        self.activation = random.uniform(0, 1)
    
    def show(self):
        print('Weights: ',self.weights)
        print('Bias: ', self.bias)
        print('Activation: ', self.activation)
"""
layer = Layer(5, None)
layer2 = Layer(3, layer)
layer2.show()
"""
print(len(trainingSet[0]))      