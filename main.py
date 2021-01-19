import pandas as pd
import numpy as np
from layer import Layer
from imgSupport import showImg


trainingExamples = pd.read_csv ('./data/mnist_train.csv')
print('Loaded training examples')

# 1D Array of all the labels to the img data or the activations of the layer 1
labels = trainingExamples.iloc[:, 0].to_numpy()

# 2D Array of all the images (28 x 28) with their grayscale values
trainX = trainingExamples.iloc[:, 1:].to_numpy()


class Network:
    # layers
    def __init__(self):
        # layer1 is the input layer
        self.layer1 = Layer(784, trainX[5]/255 , None)
        self.layer2 = Layer(16, None, self.layer1)
        self.layer3 = Layer(16, None, self.layer2)
        # layer4 is the output layer
        self.layer4 = Layer(10, None, self.layer3)
    
    def show(self):
        print(self.layer1)
        print(self.layer2)
        
    def train(self):
        print('trining')
        
    def predict(self, exampleIndex):
        self.layer1.setActivations(trainX[exampleIndex] / 255)
        self.layer2.setActivations(None)
        self.layer3.setActivations(None)
        self.layer4.setActivations(None)
        
        finalLayerActivations = network.layer4.getActivations()
        maximum = max(finalLayerActivations) 
        for i in range (10):
            if finalLayerActivations[i] == maximum:
                print('Output is ', i)
                showImg(trainX[exampleIndex])
                break
            
    def cost(self, exampleIndex):
        correctPrediction = np.array([])
        
        for i in range(10):
            if(labels[exampleIndex] == i):
                correctPrediction = np.append(correctPrediction, 1)
            else:
                correctPrediction = np.append(correctPrediction, 0)
        print(correctPrediction)
        error = np.subtract(self.layer4.getActivations(), correctPrediction)
        print(error)
        return np.dot(error, error)
            
        
        
network = Network()
network.predict(0)