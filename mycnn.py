import numpy as np
import pickle 

from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer
from utils import Utils

class MyCNN:
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def save_model(self, name):
        with open(name, 'wb') as handle:
            pickle.dump(self.layers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        with open(name, 'rb') as handle:
            b = pickle.load(handle)
        self.layers = []
        self.layers = b.copy()

    def forward(self, inputs):
        out = inputs.copy()
        for layer in self.layers:
            out = layer.forward(out)

        return out

    # Private function to forward propagate and store every output in array
    def _forward(self,inputs):
        out = inputs.copy()
        result = [out]
        for layer in self.layers:
            out = layer.forward(out)
            result.append(out)

        return result

    def fit(self, features, target, batch_size, epochs, learning_rate, momentum=1):
        for i in range(epochs):
            
            print("\rEpoch:", i, end='', flush=True)
            sum_target = 0
            sum_output = 0
            for j in range(batch_size):
                curr_index = (batch_size * i + j) % len(features) 

                # Feed forward
                result = self._forward(features[curr_index])
                dE = np.array([target[curr_index] - result[len(result)-1][0]])
                for i in reversed(range(len(self.layers))):
                    dE = self.layers[i].backward(dE)

            # Backward propagation
            # Output layer
            for i in reversed(range(len(self.layers))):
                self.layers[i].update_weights(learning_rate, momentum)
            
            
            
        print("\rEpoch:", epochs, end='', flush=True) 
