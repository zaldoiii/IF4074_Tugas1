import numpy as np


class ConvLayer:
  def __init__(self,filter_size,num_filter,input_size,num_channel,stride=1,padding=0):
    self._filter_size = filter_size
    self._num_filter = num_filter
    self._input_size = input_size
    self._num_channel = num_channel
    self._stride = stride
    self._padding = padding

    self._weights =  np.random.randn(num_filter,num_channel,filter_size,filter_size) 
    self._bias = np.zeros((num_filter))
    print(self._weights)

  def zero_padding(self, inputs):
    w, h = inputs.shape[0], inputs.shape[1]
    new_w = 2 * self._padding + w
    new_h = 2 * self._padding + h
    output = np.zeros((new_w, new_h))
    output[self._padding:w+self._padding, self._padding:h+self._padding] = inputs
    return output

  def foward(self,inputs):
    channel_size = inputs.shape[0]
    width = inputs.shape[1]+2*self._padding
    height = inputs.shape[2]+2*self._padding

    self._inputs = np.zeros((channel_size, width, height))
    for c in range(inputs.shape[0]):
        self._inputs[c,:,:] = self.zero_padding(inputs[c,:,:])

    out_width = int((width - self._filter_size)/self._stride + 1)
    out_heigth = int((height - self._filter_size)/self._stride + 1)
    print(out_heigth,out_width)
    feature_maps = np.zeros((self._num_filter, out_width, out_heigth))

    for f in range(self._num_filter):
      for w in range(out_width):
        for h in range(out_heigth):
          feature_maps[f,w,h]=np.sum(self._inputs[:, w:w+self._filter_size, h:h+self._filter_size] * self._weights[f,:,:,:]) + self._bias[f]

    return feature_maps

class PoolLayer:
  def __init__(self,filter_size,stride_size,mode):
    pass
  def foward(self,input):
    # input array of feature map -> output array of feature map
    pass

class FlattenLayer:
  def init(self):
    pass
  def foward(self,input):
    pass

class DenseLayer:
  def __init__(self,n_units,activation):
    pass
  def foward(self,input):
    pass