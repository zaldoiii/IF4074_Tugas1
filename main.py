from layers import ConvLayer
from PIL import Image
import numpy as np

if __name__ == "__main__":
    layer1 = ConvLayer(filter_size=3,num_filter=3,input_size=32,num_channel=3)
    image = Image.open('cat.2.jpg')
    print(image.format)
    print(image.size)
    print(image.mode)
    image = image.resize((300,300)) 
    # image.show()
    data = np.asarray(image)
    print(data.shape)
    data = np.transpose(data,(2,0,1))
    print(data.shape)
    data2 = layer1.foward(data)
    print(data2.shape)

    data = np.transpose(data2,(1,2,0))
    print(data.shape)

    # show the image
    # image.show()