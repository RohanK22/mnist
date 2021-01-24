import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from layer import Layer
from imgSupport import showImg


trainingExamples = pd.read_csv ('./data/mnist_train.csv')
print('Loaded training examples')

# 1D Array of all the labels to the img data or the activations of the layer 1
labels = trainingExamples.iloc[:, 0].to_numpy()

# 2D Array of all the images (28 x 28) with their grayscale values
trainX = trainingExamples.iloc[:, 1:].to_numpy()

learningRate = 0.1


class Network:
    # layers
    def __init__(self):
        # layer1 is the input layer
        self.layer1 = Layer(784, trainX[5]/255 , None)
        self.layer2 = Layer(16, None, self.layer1)
        self.layer3 = Layer(16, None, self.layer2)
        # layer4 is the output layer
        self.layer4 = Layer(10, None, self.layer3)
        
        self.layer1.layerAfter = self.layer2
        self.layer2.layerAfter = self.layer3
        self.layer3.layerAfter = self.layer4
        self.layer4.layerAfter = None
        
        self.xs = np.array([])
        self.costs = np.array([])
    
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
        nTrainingExamples = 100 # len(trainX)
        for e in range(epochs):
            for trainingExampleIndex in range(nTrainingExamples):
                print('-------------------------------------------------------')
                print('Epoch ' , e + 1, ' Example: ', trainingExampleIndex, ' %Finished: ', ((trainingExampleIndex + 1)/nTrainingExamples) * 100)
                self.predict(trainingExampleIndex)
                self.SDG(trainingExampleIndex)
                self.getAccuracy(50)
                cost = self.costOne(trainingExampleIndex)
                self.costs = np.append(self.costs, cost)
                self.xs = np.append(self.xs, trainingExampleIndex)
                plt.plot(self.xs, self.costs)
                plt.show()
        print('Finished Training ', nTrainingExamples, ' examples')
       
    def SDG(self, trainingExampleIndex):
        self.backprop(trainingExampleIndex, self.layer3)
        print('4', end=' ')
        self.backprop(trainingExampleIndex,self.layer4)
        print('3', end=' ')
        self.backprop(trainingExampleIndex, self.layer2)
        print('2', end=' ')
        self.backprop(trainingExampleIndex, self.layer1)
        print('1')
    
    def backprop(self, trainingExampleIndex, layer):
        weightsArray = layer.getWeightsArray()
        biasesArray = layer.getBiases()
        layerBefore = layer.prevLayer
        layerAfter = layer.layerAfter
        if layerBefore != None: # As layer 1 does not have any weights or biases
            if layerAfter == None: # Last Layer
                modifyWeightsBy = self.finalLayerErrorWeights(trainingExampleIndex, layer)
                modifyBiasesBy = self.finalLayerErrorBiases(trainingExampleIndex, layer)
            else:
                modifyWeightsBy = self.hiddenLayerErrorWeights(trainingExampleIndex,  layer)
                modifyBiasesBy = self.hiddenLayerErrorBiases(trainingExampleIndex, layer)
            newWeightsArray = weightsArray - (modifyWeightsBy * learningRate)
            layer.setWeightsArray(newWeightsArray)
            layer.setBiasesArray((biasesArray - modifyBiasesBy) * learningRate)
            
                
        
    def finalLayerErrorWeights(self, trainingExampleIndex, layer):
        layerBefore = layer.prevLayer
        jIndex = layer.numberOfNeurons
        kIndex = layerBefore.numberOfNeurons
        modifyWeightsBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            subArray = np.array([])
            for k in range(kIndex):
                der_Cost_wrt_weightjk = 2 * (layer.activations[j] - y[j]) * self.sigmoid(layer.zArray[j]) * layerBefore.activations[k]
                subArray = np.append(subArray, der_Cost_wrt_weightjk)
            #print(subArray.shape)
            modifyWeightsBy = np.append(modifyWeightsBy, [subArray])
        #print(modifyWeightsBy.shape)
        return modifyWeightsBy.reshape((jIndex, kIndex))
                    
    def finalLayerErrorBiases(self, trainingExampleIndex, layer):
        jIndex = layer.numberOfNeurons
        modifyBiasesBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            der_Cost_wrt_biasj = 2 * (layer.activations[j] - y[j]) * self.sigmoid(layer.zArray[j])
            modifyBiasesBy = np.append(modifyBiasesBy, der_Cost_wrt_biasj)
        return modifyBiasesBy
        
    def hiddenLayerErrorWeights(self, trainingExampleIndex, layer):
        layerBefore = layer.prevLayer
        jIndex = layer.numberOfNeurons
        kIndex = layerBefore.numberOfNeurons
        modifyWeightsBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            subArray = np.array([])
            for k in range(kIndex):
                der_Cost_wrt_weightjk = self.cal_der_C_wrt_aj(layer, j, k, y) * self.sigmoid(layer.zArray[j]) * layerBefore.activations[k]
                subArray = np.append(subArray, der_Cost_wrt_weightjk)
            modifyWeightsBy = np.append(modifyWeightsBy, subArray)
        return modifyWeightsBy.reshape((jIndex, kIndex))
    
    def cal_der_C_wrt_aj(self, layer, j, k, y):
        if layer.layerAfter == None:
            return 2 * (layer.activations[j] - y[j])
        else:
            # print(layer.layerAfter.weightsArray.shape, layer.layerAfter.numberOfNeurons)
            sum = 0
            for jj in range(layer.layerAfter.numberOfNeurons):
                # print('Accessing weightsArray of' , jj, '  ', k)
                sum += layer.layerAfter.weightsArray[jj][j] * self.sigmoid(layer.layerAfter.zArray[jj], True) * self.cal_der_C_wrt_aj(layer.layerAfter, jj, j, y)
            #print(sum)
            return sum
        
    def hiddenLayerErrorBiases(self, trainingExampleIndex, layer):
        layerBefore = layer.prevLayer
        jIndex = layer.numberOfNeurons
        kIndex = layerBefore.numberOfNeurons
        modifyBiasesBy = np.array([])
        y = self.getExpectedOutputArray(trainingExampleIndex)
        for j in range(jIndex):
            sum = 0 
            for k in range(kIndex):
                der_Cost_wrt_biasj = self.cal_der_C_wrt_aj(layer, j, k, y) * self.sigmoid(layer.zArray[j])
                sum += der_Cost_wrt_biasj
            modifyBiasesBy = np.append(modifyBiasesBy, sum)
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
                print('Final layer activations: ', finalLayerActivations)
                print('Network Prediction: ', i)
                print('Label: ', labels[exampleIndex])
                # showImg(trainX[exampleIndex])
                self.costOne(exampleIndex, True)
                return i
            
    def costOne(self, exampleIndex, printCost = False):
        correctPrediction = self.getExpectedOutputArray(exampleIndex)
        error = np.subtract(self.layer4.getActivations(), correctPrediction)
        cost = np.dot(error, error) # squared error
        if printCost == True:
            print('Cost: ', cost )
        return cost

    def getAccuracy(self, numberOfExamples, offset = 0):
        c = 0
        for i in range(numberOfExamples):
            prediction = self.predict(i + offset)
            answer = labels[i + offset]
            if prediction == answer:
                c+=1
        print('Accuracy over ', numberOfExamples, ' samples is: ', (c/numberOfExamples) * 100, '%')
    
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
network.train(1)