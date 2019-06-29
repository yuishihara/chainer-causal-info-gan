import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
from chainer.distributions import Normal, Uniform


class ImageGenerator(object):
    def __init__(self, generator, transition):
        super(ImageGenerator, self).__init__()
        self._generator = generator
        self._transition = transition

    def __call__(self, device):
        print('device id: ', device)
        xp = np if device < 0 else cp

        low = xp.ones(shape=(1), dtype=xp.float32) * -1
        high = xp.ones(shape=(1), dtype=xp.float32)
        self._Uniform = Uniform(low=low, high=high)

        loc = xp.zeros(shape=(1), dtype=xp.float32)
        scale = xp.ones(shape=(1), dtype=xp.float32)
        self._Normal = Normal(loc=loc, scale=scale)

        with chainer.no_backprop_mode():
            chainer.config.train = False

            s_current = self._Uniform.sample(sample_shape=(1, 7))
            s_current = F.squeeze(s_current)
            s_current = s_current.reshape((1, ) + s_current.shape)
            s_next, _ = self._transition(s_current)

            z = self._Normal.sample(sample_shape=(1, 4))
            z = F.squeeze(z)
            z = z.reshape((1, ) + z.shape)
            x = F.concat((z, s_current, s_next), axis=1)
            x = F.reshape(x, shape=x.shape + (1, 1))
            o = self._generator(x)
            chainer.config.train = True
            return F.split_axis(o, 2, axis=1, force_tuple=True)
