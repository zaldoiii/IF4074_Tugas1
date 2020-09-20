from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer
from mycnn import MyCNN
from PIL import Image
import numpy as np

if __name__ == "__main__":
    image = Image.open('cat.2.jpg')
    image = image.resize((300,300)) 
    # image.show()
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))
    print(data.shape)

    cnn = MyCNN()
    out = cnn.forward(data)
    print(out)
