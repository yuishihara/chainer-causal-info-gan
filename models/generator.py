from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I

import researchutils.chainer.initializers as ruinitializers


class Generator(Chain):
    def __init__(self, out_color_num=1, out_frame_num=2):
        super(Generator, self).__init__()
        with self.init_scope():
            # 1x1 -> 4x4
            self.deconv1 = L.Deconvolution2D(in_channels=None,
                                             out_channels=512,
                                             ksize=(4, 4),
                                             stride=1,
                                             nobias=True,
                                             initialW=I.Normal(scale=0.02))
            self.batch_norm1 = L.BatchNormalization(size=512,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())
            # 4x4 -> 8x8
            self.deconv2 = L.Deconvolution2D(in_channels=512,
                                             out_channels=256,
                                             ksize=(4, 4),
                                             pad=(1, 1),
                                             stride=2,
                                             nobias=True,
                                             initialW=I.Normal(scale=0.02))
            self.batch_norm2 = L.BatchNormalization(size=256,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 8x8 -> 16x16
            self.deconv3 = L.Deconvolution2D(in_channels=256,
                                             out_channels=128,
                                             ksize=(4, 4),
                                             pad=(1, 1),
                                             stride=2,
                                             nobias=True,
                                             initialW=I.Normal(scale=0.02))
            self.batch_norm3 = L.BatchNormalization(size=128,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 16x16 -> 32x32
            self.deconv4 = L.Deconvolution2D(in_channels=128,
                                             out_channels=64,
                                             ksize=(4, 4),
                                             pad=(1, 1),
                                             stride=2,
                                             nobias=True,
                                             initialW=I.Normal(scale=0.02))
            self.batch_norm4 = L.BatchNormalization(size=64,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 32x32 -> 64x64
            out_channels = out_color_num * out_frame_num
            self.deconv5 = L.Deconvolution2D(in_channels=64,
                                             out_channels=out_channels,
                                             ksize=(4, 4),
                                             pad=(1, 1),
                                             stride=2,
                                             initialW=I.Normal(scale=0.02))

    def __call__(self, x):
        # must be cx1x1
        assert x.shape[2] == 1 and x.shape[3] == 1
        h = self.deconv1(x)
        h = self.batch_norm1(h)
        h = F.relu(h)

        h = self.deconv2(h)
        h = self.batch_norm2(h)
        h = F.relu(h)

        h = self.deconv3(h)
        h = self.batch_norm3(h)
        h = F.relu(h)

        h = self.deconv4(h)
        h = self.batch_norm4(h)
        h = F.relu(h)

        h = self.deconv5(h)
        return F.tanh(h)


if __name__ == "__main__":
    import numpy as np

    test_input = np.ones(shape=(32, 7 + 7 + 4, 1, 1), dtype=np.float32)

    generator = Generator()
    image = generator(test_input)

    assert image.shape == (32, 2, 64, 64)
