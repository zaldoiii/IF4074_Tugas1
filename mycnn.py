from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer

class MyCNN:
  def __init__(self):
    self._layer1 = ConvLayer(filter_size=3,num_filter=3,input_size=32,num_channel=3)
    self._layer2 = PoolLayer(3, 1, "Max")
    self._layer3 = DetectorLayer()
    self._layer4 = FlattenLayer()
    self._layer5 = DenseLayer(n_inputs=262848, n_units=128, activation='relu')
    self._layer6 = DenseLayer(n_inputs=128, n_units=12, activation='softmax')
    self._layer7 = DenseLayer(n_inputs=12, n_units=1, activation='sigmoid')

  def forward(self,inputs):
    out = inputs.copy()
    out = self._layer1.forward(out)
    out = self._layer2.forward(out)
    out = self._layer3.forward(out)
    out = self._layer4.forward(out)
    print('input length:\n', len(out))
    out = self._layer5.forward(out)
    out = self._layer6.forward(out)
    out = self._layer7.forward(out)

    return out