import numpy as np

class ConvLayer:
  def __init__(self,filter_size,num_filter,input_size,num_channel,stride=1,padding=0):
    self._filter_size = filter_size
    self._num_filter = num_filter
    self._input_size = input_size
    self._num_channel = num_channel
    self._stride = stride
    self._padding = padding

    self._weight =  np.random.randn(num_filter,num_channel,filter_size,filter_size) 
    self._bias = np.zeros((num_filter,num_channel))
    print(self._weight)

  def foward(self,input):
    pass

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