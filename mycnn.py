from layers import ConvLayer, PoolLayer, DenseLayer, DetectorLayer, FlattenLayer

class MyCNN:
  def __init__(self):
    self._layer1 = ConvLayer(filter_size=3,num_filter=3,input_size=32,num_channel=3)
    self._layer2 = PoolLayer(3, 1, "Max")
    self._layer3 = DetectorLayer()
    # self._layer4 = FlattenLayer()

  def forward(self,inputs):
    out = inputs.copy()
    out = self._layer1.forward(out)
    out = self._layer2.forward(out)
    out = self._layer3.forward(out)
    # out = self._layer4.forward(out)

    return out