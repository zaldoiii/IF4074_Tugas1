from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer
from mycnn import MyCNN
from PIL import Image
import numpy as np

if __name__ == "__main__":
    # Cat prediction 1
    image = Image.open('cats/cat.0.jpg')
    image = image.resize((300,300))
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('cat.0.jpg prediction:\n', out)

    # Cat prediction 2
    image = Image.open('cats/cat.2.jpg')
    image = image.resize((300,300)) 
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('\n\ncat.2.jpg prediction:\n', out)

    # Cat prediction 3
    image = Image.open('cats/cat.9.jpg')
    image = image.resize((300,300))
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('\n\ncat.9.jpg prediction:\n', out)

    # Dog prediction 1
    image = Image.open('dogs/dog.0.jpg')
    image = image.resize((300,300))
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('\n\ndog.0.jpg prediction:\n', out)

    # Dog prediction 2
    image = Image.open('dogs/dog.3.jpg')
    image = image.resize((300,300))
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('\n\ndog.3.jpg prediction:\n', out)

    # Dog prediction 3
    image = Image.open('dogs/dog.4.jpg')
    image = image.resize((300,300))
    data = np.asarray(image)
    data = np.transpose(data,(2,0,1))

    cnn = MyCNN()
    out = cnn.forward(data)
    print('\n\ndog.4.jpg prediction:\n', out)
