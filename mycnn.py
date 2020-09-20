from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer

class MyCNN:
  def __init__(self):
    self._layer1 = ConvLayer(filter_size=3,num_filter=3,num_channel=3)
    self._layer2 = PoolLayer(filter_size=3,stride_size=1,mode="Max")
    self._layer3 = DetectorLayer()

    self._layer4 = ConvLayer(filter_size=4,num_filter=1,num_channel=3)
    self._layer5 = PoolLayer(filter_size=3,stride_size=2,mode="Max")
    self._layer6 = DetectorLayer()

    self._layer7 = ConvLayer(filter_size=4,num_filter=1,num_channel=1)
    self._layer8 = PoolLayer(filter_size=4,stride_size=2,mode="Max")
    self._layer9 = DetectorLayer()

    self._layer10 = FlattenLayer()
    self._layer11 = DenseLayer(n_inputs=4900, n_units=2, activation='softmax')

  def forward(self,inputs):
    out = inputs.copy()
    out = self._layer1.forward(out)
    out = self._layer2.forward(out)
    out = self._layer3.forward(out)
    out = self._layer4.forward(out)
    out = self._layer5.forward(out)
    out = self._layer6.forward(out)
    out = self._layer7.forward(out)
    out = self._layer8.forward(out)
    out = self._layer9.forward(out)
    out = self._layer10.forward(out) 
    out = self._layer11.forward(out)
    print('input length:\n', len(out))

    return out