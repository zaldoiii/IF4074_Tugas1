import math
import numpy as np


class ConvLayer:
    def __init__(self, filter_size, num_filter, input_size, num_channel, stride=1, padding=0):
        self._filter_size = filter_size
        self._num_filter = num_filter
        self._input_size = input_size
        self._num_channel = num_channel
        self._stride = stride
        self._padding = padding

        self._weights = np.random.randn(
            num_filter, num_channel, filter_size, filter_size)
        self._bias = np.zeros((num_filter))
        print(self._weights)

    def zero_padding(self, inputs):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * self._padding + w
        new_h = 2 * self._padding + h
        output = np.zeros((new_w, new_h))
        output[self._padding:w+self._padding,
               self._padding:h+self._padding] = inputs
        return output

    def foward(self, inputs):
        channel_size = inputs.shape[0]
        width = inputs.shape[1]+2*self._padding
        height = inputs.shape[2]+2*self._padding

        self._inputs = np.zeros((channel_size, width, height))
        for c in range(inputs.shape[0]):
            self._inputs[c, :, :] = self.zero_padding(inputs[c, :, :])

        out_width = int((width - self._filter_size)/self._stride + 1)
        out_heigth = int((height - self._filter_size)/self._stride + 1)
        print(out_heigth, out_width)
        feature_maps = np.zeros((self._num_filter, out_width, out_heigth))

        for f in range(self._num_filter):
            for w in range(out_width):
                for h in range(out_heigth):
                    feature_maps[f, w, h] = np.sum(
                        self._inputs[:, w:w+self._filter_size, h:h+self._filter_size] * self._weights[f, :, :, :]) + self._bias[f]

        return feature_maps


class PoolLayer:
    def __init__(self, filter_size, stride_size, mode):
        pass

    def foward(self, input):
        # input array of feature map -> output array of feature map
        pass


class FlattenLayer:
    def init(self):
        pass

    def foward(self, input):
        pass


class DenseLayer:

    bias = 1

    def __init__(self, n_units, activation):
        self.n_units = n_units
        self.activation = activation

    def _sigmoid(self, nett):
        '''
        Return the result of activation sigmoid function from a single nett
        EX.
        var nett = 3.6
        sigmoid(nett) = 1/(1+e^-nett) = 1/(1+e^(-3.6)) = 0.9734
        '''
        return 1/(1+math.e**(-nett))

    def _softmax(self, np_vector):
        '''
        Return the result of activation softmax function from nett array
        EX.
        Calculate softmax of first element 
        softmax(nett(1), nett) = e^(nett(1))/sum(e^nett(i)), i = 1, 2, 3,.., length of nett
        '''
        expo = np.exp(np_vector)
        expo_sum = np.sum(np.exp(np_vector))
        return expo/expo_sum
    
    def _nett(self, input, weight):
        '''
        Sum of input multiplied by weight
        EX. 
        var data_x = [3,2,5]
        var weight = [0.3, 0.1, 0.5, 0.7] # 0.7 is for bias
        nett(data_x, weight) = 3*0.3 + 2*0.1 + 5*0.5 + 0.7 = 4.3 
        '''

        nett_result = np.array([])

        for i in range(self.n_units):
            nett_temp = np.multiply(weight[i], np.append(input, self.bias))
            nett_result = np.append(nett_result, np.sum(nett_temp))

        print("nett:\n",nett_result)
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

    def foward(self, input, weight=[]):
        # Init weight with 0, length is equal to n_units + 1 because of bias
        if len(weight) == 0:
            weight = np.zeros( (self.n_units, len(input) + 1) )

        nett = self._nett(input, weight)
        return self._activation_function(nett)
