from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, numberOfNeurons, activations, prevLayer):
        # Number of neurons in a layer
        self.numberOfNeurons = numberOfNeurons
        # Array of neuron objects
        self.neurons = []
        self.prevLayer = prevLayer
        
        if prevLayer == None:
            numberOfWeights = 0
        else:
            numberOfWeights = prevLayer.numberOfNeurons
            
        for i in range(numberOfNeurons):
            newNeuron = Neuron(numberOfWeights)
            self.neurons.append(newNeuron)
        
        if prevLayer == None:
            self.setActivations(activations)
        else:
            self.setArrays(activations)
        
            
    def show(self):
        print('Number of Neurons: ', self.numberOfNeurons)
        print('Activations Array shape: ', self.activations.shape)
        if self.prevLayer != None:
            print('Weights Array shape: ', self.weightsArray.shape)
            print('Biases Array shape: ', self.biasesArray.shape)
        
    def setArrays(self, activations):
        # Activation Array
        self.activations = None
        self.setActivations(activations)
        
        # Weights Array
        self.weightsArray = self.getWeightsArray()
        # Biases Array
        self.biasesArray = self.getBiases()
        self.zArray = None
        self.setzArray()
        
    def setActivations(self, activations):
        # For layer 1 the activations are predefined based on the image input
        # print(type(activations))
        if isinstance(activations, np.ndarray):
            self.activations = activations
            i = 0
            for neuron in self.neurons:
                neuron.setActivation(activations[i], self.prevLayer)
                i += 1
        else:
            self.activations = np.array([])
            for neuron in self.neurons:
                self.activations = np.append(self.activations, neuron.computeActivation(self.prevLayer))
                neuron.setActivation(None, self.prevLayer)
        
    def setzArray(self):
        # For layer 1 the activations are predefined based on the image input
        # print(type(activations))
        self.zArray = np.array([])
        for neuron in self.neurons:
            self.zArray = np.append(self.zArray, neuron.computeZ(self.prevLayer))
            
    def setWeightsArray(self, weightsArray):
        self.weightsArray = weightsArray
        for j in range(len(weightsArray)):
            self.weightsArray[j] = weightsArray[j]
        
            
    # Returns a 2D array with all the weights corresponding to each neuron within the layer
    def getWeightsArray(self):
        weightsArray = np.array([])
        for neuron in self.neurons:
            weightsArray = np.append(weightsArray, neuron.weights)
        return weightsArray.reshape(len(self.neurons), len(self.neurons[0].weights))
    
    # Returns a 1D array with activatoins of all the neurons in the layer
    def getActivations(self):
        return self.activations
    
    def getBiases(self):
        biasesArray = np.array([])
        for neuron in self.neurons:
            biasesArray = np.append(biasesArray, neuron.bias)
        return biasesArray

    def setBiasesArray(self, biasesArray):
        self.biasesArray = biasesArray
        for i in range(len(self.neurons)):
            self.neurons[i].setBias(self.biasesArray[i])
        
