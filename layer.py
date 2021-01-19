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
        # Activation Array
        # Weights Array
        # Biases Array
            
    def show(self):
        print('Number of Neurons: ', self.numberOfNeurons)
        print('Neuron Objects: ', self.neurons)
        
    def getWeightsArray(self):
        weightsArray = np.array([])
        for neuron in self.neurons:
            weightsArray = np.append(weightsArray, neuron.weights)
        return weightsArray.reshape( self.prevLayer.numberOfNeurons, self.numberOfNeurons)
    
    def setActivations(self, activations):
        if activations != None:
            self.activations = activations
            i = 0
            for neuron in self.neurons:
                neuron.setActivation(self.activations[i])
                i += 1
        else:
            self.activations = np.array([])
            for neuron in self.neuron:
                self.actiavations = np.append(self.activations, neuron.getActivation(self.prevLayer))
    
    def getActivations(self):
        
        
    
            
