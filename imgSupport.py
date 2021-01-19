
from PIL import Image as im 
import numpy as np

# This function takes in a numpy array of size (784,1) and displays a (28, 28) image
def showImg(imgArray):
    imgArray = imgArray.reshape((28,28))
    img = im.fromarray(imgArray.astype(np.uint8))
    img.show()
