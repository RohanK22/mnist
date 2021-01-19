import pandas as pd
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
        self.layer1 = Layer(784, None)
        self.layer2 = Layer(16, self.layer1)
        self.layer3 = Layer(16, self.layer2)
        # layer4 is the output layer
        self.layer4 = Layer(10, self.layer3)
    
    def show(self):
        print(self.layer1)
        print(self.layer2)
        
    def train(self):
        print('trining')
        
    def predict(self):
        print('Output is ')
        

network = Network()
network.show()
