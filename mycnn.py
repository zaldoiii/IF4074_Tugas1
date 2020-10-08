import numpy as np
from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer
from utils import Utils

class MyCNN:
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
    
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
    
    def calculate_output_error(self, output, target):
        derivative_values = np.array([])
        for x in output:
            derivative_values = np.append(derivative_values, Utils.get_derivative('sigmoid', x))

        return np.multiply(derivative_values, np.subtract(target, output))

    def fit(self, features, target, batch_size, epochs, learning_rate, momentum=1):
        for i in range(epochs):
            
            print("\rEpoch:", i, end='', flush=True)
            sum_target = 0
            sum_output = 0
            for j in range(batch_size):
                curr_index = (batch_size * i + j) % len(features) 

                # Feed forward
                result = self._forward(features[curr_index])
                sum_target += target[curr_index]
                sum_output += result[len(result)-1][0] # Assume only 1 class target

            avg_target = np.array([sum_target / batch_size])
            avg_output = np.array([sum_output / batch_size])

            # Backward propagation
            # Output layer
            errors = []
            errors.append(self.calculate_output_error(avg_output, avg_target))
            curr_err_idx = 0
            # Other layers
            for i in reversed(range(len(self.layers))):
                prev_err = np.array(errors[curr_err_idx])
                errors.insert(0, self.layers[i].calculate_error(output=result[i], previous_errors=prev_err))
                curr_err_idx += 1
            
            # Update weight
            for i in reversed(range(len(self.layers))):
                self.layers[i].update_weights(np.array(errors[i+1]), result[i], learning_rate, momentum)
