from chainer import Chain
import chainer.links as L
import chainer.functions as F


class Classifier(Chain):
    def __init__(self, s_dim=7):
        super(Classifier, self).__init__()
        with self.init_scope():
            # 64x64 -> 32x32
            self.conv1 = L.Convolution2D(
                in_channels=2, out_channels=64, ksize=4, stride=2, pad=1)
            # 32x32 -> 16x16
            self.conv2 = L.Convolution2D(
                in_channels=64, out_channels=128, ksize=4, stride=2, pad=1, nobias=True)
            self.batch_norm1 = L.BatchNormalization(size=128)
            # 16x16 -> 8x8
            self.conv3 = L.Convolution2D(
                in_channels=128, out_channels=256, ksize=4, stride=2, pad=1, nobias=True)
            self.batch_norm2 = L.BatchNormalization(size=256)
            # 8x8 -> 4x4
            self.conv4 = L.Convolution2D(
                in_channels=256, out_channels=512, ksize=4, stride=2, pad=1, nobias=True)
            self.batch_norm3 = L.BatchNormalization(size=512)
            # 4x4 -> 1x1
            self.conv5 = L.Convolution2D(
                in_channels=512, out_channels=1, ksize=4)
  
    def __call__(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, slope=0.1)

        h = self.conv2(h)
        h = self.batch_norm1(h)
        h = F.leaky_relu(h, slope=0.1)

        h = self.conv3(h)
        h = self.batch_norm2(h)
        h = F.leaky_relu(h, slope=0.1)

        h = self.conv4(h)
        h = self.batch_norm3(h)
        h = F.leaky_relu(h, slope=0.1)

        h = self.conv5(h)
        return F.sigmoid(h)
