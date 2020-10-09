import math
import numpy as np

from scipy.special import expit
from utils import Utils

class ConvLayer:
    def __init__(self, filter_size, num_filter,  num_channel, stride=1, padding=0):
        self._filter_size = filter_size
        self._num_filter = num_filter
        self._num_channel = num_channel
        self._stride = stride
        self._padding = padding

        self._weights = np.random.randn(
            num_filter, num_channel, filter_size, filter_size)
        self._bias = np.zeros((num_filter))
        self._dw = np.zeros((num_filter, num_channel, filter_size, filter_size))
        self._db = np.zeros((num_filter))

    def _zero_padding(self, inputs):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * self._padding + w
        new_h = 2 * self._padding + h
        output = np.zeros((new_w, new_h))
        output[self._padding:w+self._padding,
               self._padding:h+self._padding] = inputs
        return output

    def forward(self, inputs):
        channel_size = inputs.shape[0]
        width = inputs.shape[1]+2*self._padding
        height = inputs.shape[2]+2*self._padding

        self._inputs = np.zeros((channel_size, width, height))
        for c in range(inputs.shape[0]):
            self._inputs[c, :, :] = self._zero_padding(inputs[c, :, :])

        out_width = int((width - self._filter_size)/self._stride + 1)
        out_heigth = int((height - self._filter_size)/self._stride + 1)
        feature_maps = np.zeros((self._num_filter, out_width, out_heigth))

        for f in range(self._num_filter):
            for w in range(out_width):
                for h in range(out_heigth):
                    feature_maps[f, w, h] = np.sum(
                        self._inputs[:, w:w+self._filter_size, h:h+self._filter_size] * self._weights[f, :, :, :]) + self._bias[f]

        return feature_maps

    # Reset for error
    def _reset_error(self):
        self._dw = np.zeros((self._num_filter, self._num_channel, self._filter_size, self._filter_size))
        self._db = np.zeros((self._num_filter))

    # Update weight when 1 batch is finished
    def update_weights(self,learning_rate, momentum):
        self._weights -= learning_rate * self._dw
        self._bias -= learning_rate * self._db

        self._reset_error()

    # Backprop calculation
    def backward(self, prev_errors):
        C, W, H = self._inputs.shape
        dx = np.zeros(self._inputs.shape)
        dw = np.zeros(self._weights.shape)
        db = np.zeros(self._bias.shape)

        F, W, H = prev_errors.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=prev_errors[f,w,h]*self._inputs[:,w:w+self._filter_size,h:h+self._filter_size]
                    dx[:,w:w+self._filter_size,h:h+self._filter_size]+=prev_errors[f,w,h]*self._weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(prev_errors[f, :, :])

        self._dw += dw
        self._db += db
        return dx



class PoolLayer:
    def __init__(self,filter_size,stride_size,mode):
        self._filter_size = filter_size
        self._stride_size = stride_size
        self._mode = mode

    def forward(self, inputs):
        self.input = inputs
        channel_size = inputs.shape[0]
        new_width = int((inputs.shape[1] - self._filter_size) / self._stride_size) + 1
        new_height = int((inputs.shape[2] - self._filter_size) / self._stride_size) + 1

        pooled_map = np.zeros([channel_size, new_width, new_height], dtype=np.double)

        for f in range(0, channel_size):
            for w in range(0, new_width):
                for h in range(0, new_height):
                    i = w*self._stride_size
                    j = h*self._stride_size
                    if (self._mode.lower() == 'average'):
                        pooled_map[f,w,h] = '%.3f' % np.average(inputs[f, i:(i+self._filter_size), j:(j+self._filter_size)])
                    elif (self._mode.lower() == 'max'):
                        pooled_map[f,w,h] = np.max(inputs[f, i:(i+self._filter_size), j:(j+self._filter_size)])
                    else:
                        pass

        return pooled_map
    
    def backward(self, prev_errors):
        F, W, H = self.input.shape
        dx = np.zeros(self.input.shape)

        print('prev_errors', prev_errors.shape)
        print('dx:',dx.shape)
        for f in range(0, F):
            for w in range(0, W, self._filter_size):
                for h in range(0, H, self._filter_size):
                    st = np.argmax(self.input[f, w:w+self._filter_size, h:h+self._filter_size])
                    (idx, idy) = np.unravel_index(st, (self._filter_size, self._filter_size))
                    dx[f, w+idx, h+idy] = prev_errors[f, w//self._filter_size, h//self._filter_size]
        return dx
    
    def update_weights(self, learning_rate, momentum):
        pass

class DetectorLayer:
    def __init__(self):
        pass

    def forward(self,inputs):
        # Use ReLU
        self.inputs = inputs
        inputs[inputs < 0] = 0
        return inputs
    
    def backward(self, prev_errors):
        dx = prev_errors.copy()
        dx[self.inputs < 0] = 0
        return dx
    
    def update_weights(self, learning_rate, momentum):
        pass

class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        flattened_map = inputs.flatten()
        return flattened_map

    def backward(self, prev_errors):
        return prev_errors.reshape(self.C, self.W, self.H)

    def update_weights(self, learning_rate, momentum):
        pass

class DenseLayer:

    def __init__(self, n_units, activation, n_inputs=None):
        self.n_units = n_units
        self.activation = activation

        self.bias = np.random.uniform(low=0.0, high=0.1, size=n_units)
        if n_inputs:
            self._init_weights(n_inputs)
        else:
            self.weight = []
            self.n_inputs = n_inputs

    def _init_weights(self, n_inputs):
        self.n_inputs = n_inputs
        # Init weight from interval 0..0.1
        self.weight = np.random.uniform(low=0.0, high=0.1, size=(self.n_units, n_inputs))
        self.deltaW = np.random.uniform(low=0.0, high=0.1, size=self.n_units)

    def _sigmoid(self, nett):
        '''
        Return the result of activation sigmoid function from a single nett
        EX.
        var nett = 3.6
        sigmoid(nett) = 1/(1+e^-nett) = 1/(1+e^(-3.6)) = 0.9734
        '''
        return 1/(1 + np.exp(-nett))

    def _softmax(self, np_vector):
        '''
        Return the result of activation softmax function from nett array
        EX.
        Calculate softmax of first element 
        softmax(nett(1), nett) = e^(nett(1))/sum(e^nett(i)), i = 1, 2, 3,.., length of nett
        '''
        expo = expit(np_vector)
        expo_sum = np.sum(expit(np_vector))
        return expo/expo_sum
    
    def _nett(self, input):
        '''
        Sum of input multiplied by weight
        EX. 
        var input = [3,2,5]
        var weight = [0.3, 0.1, 0.5]
        var bias = 0.7
        nett(input) = 3*0.3 + 2*0.1 + 5*0.5 + 0.7 = 4.3 
        '''

        nett_result = np.array([])

        for i in range(self.n_units):
            nett_temp = np.multiply(self.weight[i], input)
            nett_result = np.append(nett_result, np.sum(nett_temp) + self.bias[i] )

        return nett_result

    def _activation_function(self, nett):
        if self.activation == 'sigmoid':
            sigmoid_v = np.vectorize(self._sigmoid)
            return sigmoid_v(nett)
        elif self.activation == 'relu':
            return np.maximum(nett, 0)
        elif self.activation == 'softmax':
            return self._softmax(nett)
        else:
            raise Exception("Undefined activation function")

    def forward(self, inputs):
        if len(self.weight) == 0:
           self._init_weights(len(inputs))
        self.input = inputs

        nett = self._nett(inputs)
        self.output = self._activation_function(nett)
        return self.output

    # Reset for error
    def _reset_error(self):
        self.deltaW = np.random.uniform(low=0.0, high=0.1, size=self.n_units)

    # Update weights and bias
    def update_weights(self, learning_rate, momentum):
        # Update weight formula = w + momentum * w + learning_rate * errors * output
        # Update bias formula = bias + momentum * bias + learning_rate * errors
        for i in range(self.n_units):
            self.weight[i] = self.weight[i] + ((momentum * self.weight[i]) + (learning_rate * self.deltaW[i] * self.input))

        self.bias = self.bias + ((momentum * self.bias) + (learning_rate * self.deltaW))

        self._reset_error()
    
    # Update weight in the current layers based on previous errors and calculate error for the current network
    def backward(self, prev_errors):
        derivative_values = np.array([])
        for x in self.output:
            derivative_values = np.append(derivative_values, Utils.get_derivative(self.activation, x))

        self.deltaW += np.multiply(derivative_values, prev_errors)
        # weight matrix representation: row for output, column for input
        dE = np.matmul(prev_errors, self.weight)

        return dE