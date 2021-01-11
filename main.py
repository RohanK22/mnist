import pandas as pd

trainingExamples = pd.read_csv ('./data/mnist_train.csv')
print('Loaded training examples')

# 1D Array of all the labels to the img data 
labels = trainingExamples.iloc[:, 0].to_numpy()

# 2D Array of all the images (28 x 28) with their grayscale values
images = trainingExamples.iloc[:, 1:].to_numpy()
print (type(images[0]))