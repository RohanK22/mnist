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
        print('Layer 1 details: ')
        self.layer1.show()
        print('Layer 2 details: ')
        self.layer2.show()
        print('Layer 3 details: ')
        self.layer3.show()
        print('Layer 4 details: ')
        self.layer4.show()
        
    def train(self, epochs):
        print('Training Started')
        for e in range(epochs):
            for trainingExampleIndex in range(len(trainX)):
                self.SDG(trainingExampleIndex)
        
    def SDG(self, trainingExampleIndex):
        self.backprop(trainingExampleIndex, self.layer3, self.layer4, None)
        self.backprop(trainingExampleIndex, self.layer2, self.layer3, self.layer4)
        self.backprop(trainingExampleIndex, self.layer1, self.layer2, self.layer3)
        self.backprop(trainingExampleIndex, None,        self.layer1, self.layer2)
    
    def backprop(self, trainingExampleIndex, layerBefore, layer, layerAfter):
        if layerBefore != None: # As layer 1 does not have any weights or biases
            if layerAfter == None: # Last Layer
                weightsArray = layer.getWeightsArray()
                biasesArray = layer.getBiases()
                layer.setWeightsArray(weightsArray - self.finalLayerErrorWeights(trainingExampleIndex, layerBefore.numberOfNeurons, layer.numberOfNeurons, layerBefore, layer, layerAfter))
                layer.setBiasesArray(biasesArray - self.finalLayerErrorBiases(trainingExampleIndex, layerBefore.numberOfNeurons, layer.numberOfNeurons, layerBefore, layer, layerAfter))
                # Recompute all the activations of the last layer
                layer.setActivations()
            else:
                
                
        
    def finalLayerErrorWeights(self, trainingExampleIndex, jIndex, kIndex, layerBefore, layer, layerAfter):
        modifyWeightsBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            subArray = np.array([])                
            for k in range(kIndex):
                sum = 2 * (layer.activations[k] - y[k]) * self.sigmoid(layer.zArray[k]) * layerBefore.activations[j]
                subArray = np.append(subArray, sum)
            modifyWeightsBy = np.append(modifyWeightsBy, subArray)
        return modifyWeightsBy
                    
    def finalLayerErrorBiases(self, trainingExampleIndex, jIndex, kIndex, layerBefore, layer, layerAfter):
        modifyBiasesBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            subArray = np.array([])                
            for k in range(kIndex):
                sum = 2 * (layer.activations[k] - y[k]) * self.sigmoid(layer.zArray[k])
                subArray = np.append(subArray, sum)
            modifyBiasesBy = np.append(modifyBiasesBy, subArray)
        return modifyBiasesBy
    
    def predict(self, exampleIndex):
        self.layer1.setActivations(trainX[exampleIndex] / 255)
        self.layer2.setActivations(None)
        self.layer3.setActivations(None)
        self.layer4.setActivations(None)
        
        finalLayerActivations = network.layer4.getActivations()
        maximum = max(finalLayerActivations) 
        for i in range (10):
            if finalLayerActivations[i] == maximum:
                print('Network Prediction: ', i)
                print('Label: ', labels[exampleIndex])
                # showImg(trainX[exampleIndex])
                break
            
    def costOne(self, exampleIndex):
        correctPrediction = self.getExpectedOutputArray(exampleIndex)
        error = np.subtract(self.layer4.getActivations(), correctPrediction)
        cost = np.dot(error, error) # squared error
        print('cost: ', cost )
        return cost
    
    def getExpectedOutputArray(self, exampleIndex):
        correctPredictionArray = np.array([])
        for i in range(10):
            if(labels[exampleIndex] == i):
                correctPredictionArray = np.append(correctPredictionArray, 1)
            else:
                correctPredictionArray = np.append(correctPredictionArray, 0)
        return correctPredictionArray
        
    def sigmoid(self, x, derivative=False):
        if (derivative == True):
            return self.sigmoid(x,derivative=False) * (1 - self.sigmoid(x,derivative=False))
        else:
            return 1 / (1 + np.exp(-x))
        
        
network = Network()
network.show()
network.predict(0)
network.costOne(0)