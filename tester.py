import os
import argparse

import cv2

import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
from chainer import initializers
from chainer import optimizers
from chainer.dataset import to_device
from chainer.distributions import Normal

import researchutils.chainer.serializers as serializers
from researchutils import files
from researchutils.image import viewer
from researchutils.image import converter

from models.generator import Generator
from models.posterior import Posterior
from models.transition import Transition
from models.subtractor import Subtractor
from models.classifier import Classifier

from image_generator import ImageGenerator


class OptimizableLatentState(chainer.Link):
    def __init__(self, s_shape, z_shape):
        super(OptimizableLatentState, self).__init__()
        with self.init_scope():
            self._s = chainer.Parameter(
                initializers.Uniform(scale=1), s_shape)
            self._z = chainer.Parameter(
                initializers.Normal(scale=0.01), z_shape)

    def __call__(self):
        return self._s, self._z


def prepare_generator(args):
    generator = Generator()
    assert files.file_exists(args.generator_parameter)
    serializers.load_model(args.generator_parameter, generator)
    return generator


def prepare_posterior(args):
    posterior = Posterior()
    assert files.file_exists(args.posterior_parameter)
    serializers.load_model(args.posterior_parameter, posterior)
    return posterior


def prepare_transition(args):
    transition = Transition()
    assert files.file_exists(args.transition_parameter)
    serializers.load_model(args.transition_parameter, transition)
    return transition


def prepare_subtractor(args):
    subtractor = Subtractor()
    assert files.file_exists(args.subtractor_parameter)
    serializers.load_model(args.subtractor_parameter, subtractor)
    return subtractor


def prepare_classifier(args):
    classifier = Classifier()
    assert files.file_exists(args.classifier_parameter)
    serializers.load_model(args.classifier_parameter, classifier)
    return classifier


def load_image(image_file):
    print('loading images from: ', image_file)
    image = cv2.imread(image_file)
    image = cv2.resize(image, dsize=(64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = converter.hwc2chw(image) / 255.0
    image = image.astype(np.float32)
    assert np.all(0.0 <= image) and np.all(image <= 1.0)
    return image


def subtract_background(subtractor, image):
    subtracted = subtractor(image)
    subtracted = chainer.Variable(subtracted.data)
    return F.clip(2 * (subtracted - 0.5), -1.0 + 1e-3, 1.0 - 1e-3)


def find_closest_latent_state(real_o, generator, transition, classifier, args):
    trials = 400
    target = OptimizableLatentState(s_shape=(trials, 7), z_shape=(trials, 4))
    if not args.gpu < 0:
        target.to_gpu()

    _, channels, height, width = real_o.shape
    real_o = real_o.reshape((channels, height, width))
    real_o = F.broadcast_to(real_o, (trials, ) + real_o.shape)
    print('real_o shape: ', real_o.shape)

    optimizer = optimizers.Adam(alpha=1e-2)
    optimizer.setup(target)

    iterations = 1000

    def compute_loss(real_o, o_current):
        concat_image = F.concat((real_o, o_current),  axis=1)
        classification_loss = classifier(concat_image)
        classification_loss = F.squeeze(classification_loss)
        l2_loss = F.batch_l2_norm_squared(real_o - o_current)
        assert classification_loss.shape == l2_loss.shape
        loss = l2_loss - classification_loss
        return loss

    s_current, z = target()
    for i in range(iterations):
        optimizer.target.cleargrads()

        s_next, _ = transition(s_current)
        # print('s_current shape: ', s_current.shape, 's_next shape: ', s_next.shape)
        x = F.concat((z, s_current, s_next), axis=1)
        x = F.reshape(x, shape=x.shape + (1, 1))
        o = generator(x)
        o_current, _ = F.split_axis(o, 2, axis=1, force_tuple=True)
        # print('o shape: ', o_current.shape)
        # print('real_o shape: ', real_o.shape)

        loss = compute_loss(real_o, o_current)
        mean_loss = F.mean(loss)
        mean_loss.backward()
        optimizer.update()
        mean_loss.unchain_backward()

        if i % 100 == 0:
            index = F.argmin(loss).data
            print('loss at: ', i, ' min index: ', index,
                  ' min loss: ', loss[index])

    # Select s and z with min loss
    s_current, z = target()
    s_next, _ = transition(s_current)
    x = F.concat((z, s_current, s_next), axis=1)
    x = F.reshape(x, shape=x.shape + (1, 1))
    o = generator(x)
    o_current, _ = F.split_axis(o, 2, axis=1, force_tuple=True)
    loss = compute_loss(real_o, o_current)

    index = F.argmin(loss).data
    print('min index: ', index, ' min loss: ', loss[index])

    s_min = s_current.data[index]
    print('s min: ', s_min)
    z_min = z.data[index]
    print('z min: ', z_min)
    return chainer.Variable(s_min), chainer.Variable(z_min)


def predict(start_image, goal_image, generator, posterior, transition, subtractor, classifier, args):
    start = to_device(device=args.gpu, x=start_image.reshape(
        (1, ) + start_image.shape))
    start = subtract_background(subtractor, start)
    goal = to_device(device=args.gpu, x=goal_image.reshape(
        (1, ) + goal_image.shape))
    goal = subtract_background(subtractor, goal)
    print('start image shape: ', start.shape)
    print('goal image shape: ', goal.shape)

    xp = np if args.gpu < 0 else cp

    loc = xp.zeros(shape=(1), dtype=np.float32)
    scale = xp.ones(shape=(1), dtype=np.float32)
    _Normal = Normal(loc=loc, scale=scale)

    sequence = []
    chainer.config.train = False

    start_state, _ = find_closest_latent_state(
        start, generator, transition, classifier, args)
    goal_state, _ = find_closest_latent_state(
        goal, generator, transition, classifier, args)

    with chainer.no_backprop_mode():
        start_state = start_state.reshape((1, ) + start_state.shape)
        goal_state = goal_state.reshape((1, ) + goal_state.shape)

        s_current = start_state
        print('current_state: ', s_current.shape)
        step = (goal_state - start_state) / args.steps
        z = _Normal.sample(sample_shape=(1, 4))
        z = F.squeeze(z)
        z = z.reshape((1, ) + z.shape)
        for i in range(1, args.steps):
            s_next = start_state + step * i
            x = F.concat((z, s_current, s_next), axis=1)
            x = F.reshape(x, shape=x.shape + (1, 1))
            o = generator(x)
            _, o_next = F.split_axis(o, 2, axis=1, force_tuple=True)

            o_next.to_cpu()
            sequence.append(o_next.data[0][0])
            s_current = s_next
    return sequence


def show_sequence(images):
    viewer.animate(images=images, is_gray=True, save_mp4=True, save_gif=True)


def show_image_sample(generator, transition):
    image_generator = ImageGenerator(generator, transition)

    image_current, image_next = image_generator(generator.device.device.id)

    image_current.to_cpu()
    image_next.to_cpu()

    viewer.show_image(image_current.data[0][0], is_gray=True)
    viewer.show_image(image_next.data[0][0], is_gray=True)


def predict_sequence(args):
    generator = prepare_generator(args)
    posterior = prepare_posterior(args)
    transition = prepare_transition(args)
    subtractor = prepare_subtractor(args)
    classifier = prepare_classifier(args)
    if not args.gpu < 0:
        generator.to_gpu()
        posterior.to_gpu()
        transition.to_gpu()
        subtractor.to_gpu()
        classifier.to_gpu()

    if args.show_transition_sample:
        show_image_sample(generator, transition)

    start_image = load_image(args.start_image)
    goal_image = load_image(args.goal_image)

    image_sequence = predict(start_image, goal_image,
                             generator, posterior, transition,
                             subtractor, classifier, args)

    show_sequence(image_sequence)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start-image', type=str, required=True)
    parser.add_argument('--goal-image', type=str, required=True)

    # parameter files
    parser.add_argument('--generator-parameter', type=str, required=True)
    parser.add_argument('--posterior-parameter', type=str, required=True)
    parser.add_argument('--subtractor-parameter', type=str, required=True)
    parser.add_argument('--transition-parameter', type=str, required=True)
    parser.add_argument('--classifier-parameter', type=str, required=True)

    # interpolation settings
    parser.add_argument('--steps', type=int, default=10)

    # gpu setting
    parser.add_argument('--gpu', type=int, default=0)

    # optional debug options
    parser.add_argument('--show-transition-sample', action='store_true')

    args = parser.parse_args()

    # show_dataset(args.dataset_dir)
    predict_sequence(args)


if __name__ == "__main__":
    main()
