from chainer import Chain
import chainer.links as L
import chainer.functions as F


class Transition(Chain):
    def __init__(self, s_dim=7):
        super(Transition, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(in_size=s_dim, out_size=64)
            self.linear2 = L.Linear(in_size=64, out_size=64)
            self.linear3 = L.Linear(in_size=64, out_size=s_dim * 2)

    def __call__(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)
        h = self.linear3(h)
        mu, ln_var = F.split_axis(h, 2, axis=-1)
        return mu, ln_var


if __name__ == "__main__":
    import numpy as np

    test_input = np.ones(shape=(32, 7), dtype=np.float32)

    transition = Transition()
    mu, ln_var = transition(test_input)

    assert mu.shape == (32, 7)
    assert ln_var.shape == (32, 7)
