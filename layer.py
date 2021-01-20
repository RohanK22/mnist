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
        
        self.setActivations(activations)
        # Activation Array
        # Weights Array
        # Biases Array
            
    def show(self):
        print('Number of Neurons: ', self.numberOfNeurons)
        print('Neuron Objects: ', self.neurons)
        
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
            
    def setWeightsArray(self, weightsArray):
        for j in weightsArray:
            for k in j:
                self.neurons[j].weights = k
        
            
    # Returns a 2D array with all the weights corresponding to each neuron within the layer
    def getWeightsArray(self):
        weightsArray = np.array([])
        for neuron in self.neurons:
            weightsArray = np.append(weightsArray, neuron.weights)
        return weightsArray.reshape( self.prevLayer.numberOfNeurons, self.numberOfNeurons)
    
    # Returns a 1D array with activatoins of all the neurons in the layer
    def getActivations(self):
        return self.activations
    
    def getBiases(self):
        biasesArray = np.array([])
        for neuron in self.neurons:
            biasesArray = np.append(biasesArray, neuron.bias)
        return biasesArray
        
