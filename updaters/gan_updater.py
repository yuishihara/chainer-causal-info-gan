import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
from chainer import training
from chainer.dataset import convert
from chainer.distributions import Uniform
from chainer.distributions import Normal

from researchutils import arrays

class GANUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer, subtractor, device,
                 mutual_info_loss_weight=0.1,
                 transition_loss_weight=0.1):
        super(GANUpdater, self).__init__(
            iterator=iterator, optimizer=optimizer, device=device)
        # Must have 1 optimizers for each network
        assert len(optimizer) == 4

        self._subtractor = subtractor

        xp = np if device < 0 else cp

        low = xp.ones(shape=(1), dtype=np.float32) * -1
        high = xp.ones(shape=(1), dtype=np.float32)
        self._Uniform = Uniform(low=low, high=high)

        loc = xp.zeros(shape=(1), dtype=np.float32)
        scale = xp.ones(shape=(1), dtype=np.float32)
        self._Normal = Normal(loc=loc, scale=scale)

        label_shape = (iterator.batch_size, 1)
        self._zero_labels = self._generate_zero_labels(
            shape=label_shape, device=device)
        self._one_labels = self._generate_one_labels(
            shape=label_shape, device=device)

        self._mutual_info_loss_weight = mutual_info_loss_weight
        self._transition_loss_weight = transition_loss_weight

    def update_core(self):
        d_optimizer = self.get_optimizer('d_optimizer')
        g_optimizer = self.get_optimizer('g_optimizer')
        p_optimizer = self.get_optimizer('p_optimizer')
        t_optimizer = self.get_optimizer('t_optimizer')

        discriminator = d_optimizer.target
        generator = g_optimizer.target
        posterior = p_optimizer.target
        transision = t_optimizer.target

        iterator = self.get_iterator('main')
        batch_size = iterator.batch_size
        batch = iterator.next()

        in_arrays = convert._call_converter(self.converter, batch, self.device)

        real_o_current = self._subtract_background(in_arrays[0])
        real_o_next = self._subtract_background(in_arrays[1])
        assert len(real_o_current) == iterator.batch_size
        assert len(real_o_next) == iterator.batch_size

        # Update discriminator network
        d_optimizer.target.cleargrads()

        real_loss = self._discriminator_loss(
            discriminator, real_o_current, real_o_next, self._one_labels)

        s_current, s_next, z = self._sample_state(
            transision, s_shape=(batch_size, 7), z_shape=(batch_size, 4))
        fake_o_current, fake_o_next = self._generate_observation(
            generator, z, s_current, s_next)
        fake_d_loss = self._discriminator_loss(
            discriminator, 
            chainer.Variable(fake_o_current.data), 
            chainer.Variable(fake_o_next.data), 
            self._zero_labels)

        d_loss = real_loss + fake_d_loss
        d_loss.backward()
        d_optimizer.update()

        # Update generator, posterior and transision network
        g_optimizer.target.cleargrads()
        p_optimizer.target.cleargrads()
        t_optimizer.target.cleargrads()

        fake_g_loss = self._discriminator_loss(
            discriminator, fake_o_current, fake_o_next, self._one_labels)

        q_current_nll = self._posterior_nll(
            posterior, fake_o_current, s_current)
        q_next_nll = self._posterior_nll(posterior, fake_o_next, s_next)
        t_pll = self._transition_pll(transision, s_current, s_next)
        transition_loss = self._transition_loss(transision, s_current)

        # NOTE: q_current_nll and q_next_nll is negative value
        mutual_information_loss = q_current_nll + q_next_nll + t_pll
        assert fake_g_loss is not None
        assert mutual_information_loss is not None
        assert transition_loss is not None
        gpt_loss = fake_g_loss + \
            self._mutual_info_loss_weight * mutual_information_loss + \
            self._transition_loss_weight * transition_loss

        gpt_loss.backward()

        g_optimizer.update()
        p_optimizer.update()
        t_optimizer.update()

        # remove backward references
        d_loss.unchain_backward()
        gpt_loss.unchain_backward()

        chainer.reporter.report({'d_loss': d_loss})
        chainer.reporter.report({'gpt_loss': gpt_loss})

    def _generate_zero_labels(self, shape, device=-1):
        labels = np.zeros(shape=shape, dtype=np.int32)
        labels = chainer.Variable(labels)
        labels.to_gpu(device=device)
        return labels

    def _generate_one_labels(self, shape, device=-1):
        labels = np.ones(shape=shape, dtype=np.int32)
        labels = chainer.Variable(labels)
        labels.to_gpu(device=device)
        return labels

    def _generate_observation(self, generator, z, s_current, s_next):
        x = F.concat((z, s_current, s_next), axis=1)
        x = F.reshape(x, shape=x.shape + (1, 1))
        o = generator(x)
        o_current, o_next = F.split_axis(o, 2, axis=1, force_tuple=True)
        return o_current, o_next

    def _posterior_nll(self, posterior, o, s):
        mu, ln_var = posterior(o)
        return F.gaussian_nll(x=s, mean=mu, ln_var=ln_var, reduce='mean')

    # pll stands for positive log likelihood
    def _transition_pll(self, transition, s_current, s_next):
        mu, ln_var = transition(s_current)
        return -F.gaussian_nll(x=s_next, mean=mu, ln_var=ln_var, reduce='mean')

    def _discriminator_loss(self, discriminator, o_current, o_next, label):
        x = F.concat((o_current, o_next), axis=1)
        p = discriminator(x)
        return F.sigmoid_cross_entropy(p, label)

    def _transition_loss(self, transition, s):
        _, ln_var = transition(s)
        var = F.exp(ln_var)
        l2_norm = F.batch_l2_norm_squared(var)
        return F.mean(l2_norm)

    def _sample_state(self, transision, s_shape=(32, 7), z_shape=(32, 4)):
        s_current = self._Uniform.sample(sample_shape=s_shape)
        s_current = F.squeeze(s_current)
        s_next, _ = transision(s_current)
        z = self._Normal.sample(sample_shape=z_shape)
        z = F.squeeze(z)
        assert s_shape == s_current.shape
        assert s_shape == s_next.shape
        assert z_shape == z.shape
        return s_current, s_next, z

    def _subtract_background(self, rgb_image):
        subtracted = self._subtractor(rgb_image)
        subtracted = chainer.Variable(subtracted.data)
        return F.clip(2 * (subtracted - 0.5), -1.0 + 1e-3, 1.0 - 1e-3) 


if __name__ == "__main__":
    # check_sigmoid_cross_entropy

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    test_input = np.ones(shape=(32, 1), dtype=np.float32)
    test_label_one = np.ones(shape=(16, 1), dtype=np.int32)
    test_label_zero = np.zeros(shape=(16, 1), dtype=np.int32)
    test_label = np.concatenate((test_label_one, test_label_zero))

    expected = -np.mean(test_label * np.log(sigmoid(test_input)) +
                        (1 - test_label) * np.log(1 - sigmoid(test_input)))
    actual = F.sigmoid_cross_entropy(test_input, test_label)

    print('test input shape: ', test_input.shape, ' bce shape: ', actual.shape)
    print('test label one shape: ', test_label_one.shape,
          ' test label zero shape: ', test_label_zero.shape)
    print('test label shape: ', test_label.shape)
    print('actual bce: ', actual, ' expected bce: ', expected)
