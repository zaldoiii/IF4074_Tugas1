from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer
from mycnn import MyCNN
from PIL import Image
import numpy as np

if __name__ == "__main__":
    layer1 = ConvLayer(filter_size=3,num_filter=3,input_size=32,num_channel=3)
    image = Image.open('cat.2.jpg')
    image = image.resize((300,300)) 
    # image.show()
    data = np.asarray(image)
    print(data.shape)
    data = np.transpose(data,(2,0,1))
    print(data.shape)

    cnn = MyCNN()
    out = cnn.forward(data)
    print(out)
    # data2 = layer1.forward(data)
    # print(data2.shape)

    # data = np.transpose(data2,(1,2,0))
    # print(data.shape)
    # print(data)

    # layer2 = DetectorLayer()
    # data = layer2.forward(data)
    # print(data)

    # # show the image
    # # image.show()

    # # POOLING LAYER
    # print("\n\nPOOLING LAYER:\n")
    # layer3 = PoolLayer(3, 1, "Max") # Bisa diganti antara 'max' atau 'average'
    # dataPooling = np.random.randint(0, 256, size=(3, 5, 5))
    # print("\nCoba feature map random:\n")
    # print(dataPooling)
    # poolingOutputTest = layer3.forward(dataPooling)
    # print("\nHasil Pooling:\n")
    # print(poolingOutputTest)

    # # DENSE LAYER
    # print("\n\nDense Layer\n\n")
    # n_units = 2
    # data_size = 5
    # flat_data = np.random.uniform(low=-10, high=10, size=data_size)
    # weight = np.random.random((n_units, data_size + 1))
    # print("Data input:\n", flat_data)
    # print("Init Weight:\n", weight)
    # layer2 = DenseLayer(n_units=n_units, activation='sigmoid')
    # out1 = layer2.forward(flat_data, weight)
    # print("Data output:\n", out1)