from sklearn import datasets

from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer
from mycnn import MyCNN
from PIL import Image
import numpy as np

# Cat prediction 2
# image = Image.open('cats/cat.0.jpg')
# image = image.resize((300,300)) 
# data = np.asarray(image)
# data = np.transpose(data,(2,0,1))

# cnn = MyCNN (
#     ConvLayer(filter_size=3,num_filter=3,num_channel=3),
#     DetectorLayer(),
#     PoolLayer(filter_size=3,stride_size=1,mode="Max"),

#     ConvLayer(filter_size=4,num_filter=1,num_channel=3),
#     DetectorLayer(),
#     PoolLayer(filter_size=3,stride_size=2,mode="Max"),

#     ConvLayer(filter_size=4,num_filter=1,num_channel=1),
#     DetectorLayer(),
#     PoolLayer(filter_size=4,stride_size=2,mode="average"),

#     FlattenLayer(),
#     DenseLayer(n_inputs=4900, n_units=120, activation='relu'),
#     DenseLayer(n_inputs=120, n_units=1, activation='sigmoid'),
# )

iris = datasets.load_iris()
X = iris.data
y = iris.target
# out = cnn.forward(data)
# print('\n\ncat.0.jpg prediction:\n', out)
cnn = MyCNN (
    DenseLayer(n_units=10, activation='sigmoid'),
    DenseLayer(n_units=3, activation='relu'),
    DenseLayer(n_units=1, activation='sigmoid'),
)

cnn.fit(
    features=X,
    target=y,
    batch_size=5,
    epochs=100,
    learning_rate=0.1
)
