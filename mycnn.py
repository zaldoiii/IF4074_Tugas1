import numpy as np
import pickle 

from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer
from sklearn import metrics
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

    def calculate_loss(self, target, output):
        return 0.5 * (target-output)**2

    def fit(self, features, target, batch_size, epochs, learning_rate, momentum=1):
        out = np.array([])
        y_target = np.array([])
        for i in range(epochs):
            
            print("Epoch:", i+1, end=', ')
            sum_loss = 0
            for j in range(batch_size):
                curr_index = (batch_size * i + j) % len(features) 

                # Feed forward
                result = self._forward(features[curr_index])
                curr_target = target[curr_index]
                curr_output = result[len(result)-1][0]
                dE = np.array([curr_target - curr_output])*-1
                for i in reversed(range(len(self.layers))):
                    dE = self.layers[i].backward(dE)
                sum_loss += self.calculate_loss(curr_target, curr_output)
                out = np.rint(np.append(out, curr_output))
                y_target = np.append(y_target, curr_target)

            # Backward propagation
            # Output layer
            for i in reversed(range(len(self.layers))):
                self.layers[i].update_weights(learning_rate, momentum)
            avg_loss = sum_loss/batch_size
            print('Loss: ', avg_loss, end=', ')
            print('Accuracy: ', metrics.accuracy_score(y_target, out))
            