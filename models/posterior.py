from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I

import researchutils.chainer.initializers as ruinitializers


class Posterior(Chain):
    def __init__(self, in_color_num=1, in_frame_num=1, c_dim=7):
        super(Posterior, self).__init__()
        with self.init_scope():
            # 64x64 -> 32x32
            in_channels = in_color_num * in_frame_num
            self.conv1 = L.Convolution2D(in_channels=in_channels,
                                         out_channels=64,
                                         ksize=(4, 4),
                                         pad=(1, 1),
                                         stride=2,
                                         nobias=True,
                                         initialW=I.Normal(scale=0.02))
            self.batch_norm1 = L.BatchNormalization(size=64,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())
            # 32x32 -> 16x16
            self.conv2 = L.Convolution2D(in_channels=64,
                                         out_channels=128,
                                         ksize=(4, 4),
                                         pad=(1, 1),
                                         stride=2,
                                         nobias=True,
                                         initialW=I.Normal(scale=0.02))
            self.batch_norm2 = L.BatchNormalization(size=128,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 16x16 -> 8x8
            self.conv3 = L.Convolution2D(in_channels=128,
                                         out_channels=256,
                                         ksize=(4, 4),
                                         pad=(1, 1),
                                         stride=2,
                                         nobias=True,
                                         initialW=I.Normal(scale=0.02))
            self.batch_norm3 = L.BatchNormalization(size=256,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 8x8 -> 4x4
            self.conv4 = L.Convolution2D(in_channels=256,
                                         out_channels=512,
                                         ksize=(4, 4),
                                         pad=(1, 1),
                                         stride=2,
                                         nobias=True,
                                         initialW=I.Normal(scale=0.02))
            self.batch_norm4 = L.BatchNormalization(size=512,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            # 4x4 -> 1x1
            self.conv5 = L.Convolution2D(in_channels=512,
                                         out_channels=128,
                                         ksize=(4, 4),
                                         stride=1,
                                         nobias=True,
                                         initialW=I.Normal(scale=0.02))
            self.batch_norm5 = L.BatchNormalization(size=128,
                                                    initial_gamma=ruinitializers.NormalWithLoc(
                                                        loc=1.0, scale=0.02),
                                                    initial_beta=I.Zero())

            self.conv_mu = L.Convolution2D(in_channels=128,
                                           out_channels=c_dim,
                                           ksize=(1, 1),
                                           stride=1,
                                           initialW=I.Normal(scale=0.02))
            self.conv_var = L.Convolution2D(in_channels=128,
                                            out_channels=c_dim,
                                            ksize=(1, 1),
                                            stride=1,
                                            initialW=I.Normal(scale=0.02))

    def __call__(self, x):
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = F.leaky_relu(h, slope=0.2)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = F.leaky_relu(h, slope=0.2)

        h = self.conv3(h)
        h = self.batch_norm3(h)
        h = F.leaky_relu(h, slope=0.2)

        h = self.conv4(h)
        h = self.batch_norm4(h)
        h = F.leaky_relu(h, slope=0.2)

        h = self.conv5(h)
        h = self.batch_norm5(h)
        # Slope is set to 0.1 see
        # https://github.com/thanard/causal-infogan/
        h = F.leaky_relu(h, slope=0.1)

        mu = self.conv_mu(h)
        mu = F.squeeze(mu)
        ln_var = self.conv_var(h)
        ln_var = F.squeeze(ln_var)
        return mu, ln_var


if __name__ == "__main__":
    import numpy as np

    test_input = np.ones(shape=(32, 1, 64, 64), dtype=np.float32)

    posterior = Posterior()
    mu, ln_var = posterior(test_input)

    assert mu.shape == (32, 7)
    assert ln_var.shape == (32, 7)
