from neuron import Neuron

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
