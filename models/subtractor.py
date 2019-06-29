from chainer import Chain
import chainer.links as L
import chainer.functions as F


class Subtractor(Chain):
    def __init__(self, s_dim=7):
        super(Subtractor, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3, out_channels=16, ksize=5, pad=2)
            self.conv2 = L.Convolution2D(
                in_channels=16, out_channels=32, ksize=5, pad=2)
            self.classifier = L.Convolution2D(
                in_channels=32, out_channels=1, ksize=1)

    def __call__(self, x):
        h = self.conv1(x)
        h = F.tanh(h)
        h = self.conv2(h)
        h = F.tanh(h)
        return self.classifier(h)
