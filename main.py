import os
import argparse

import cv2

import numpy as np

from chainer import training
from chainer import optimizers

from chainer.dataset import to_device
from chainer.datasets import tuple_dataset
from chainer import iterators

from researchutils import files
from researchutils.image import viewer
from researchutils.image import converter
import researchutils.chainer.serializers as serializers

from models.discriminator import Discriminator
from models.generator import Generator
from models.posterior import Posterior
from models.subtractor import Subtractor
from models.transition import Transition

from updaters.gan_updater import GANUpdater

from image_generator import ImageGenerator

from matplotlib import pyplot


def setup_adam_optimizer(model, lr, beta1, beta2):
    optimizer = optimizers.Adam(
        alpha=lr, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def prepare_subtractor(args):
    subtractor = Subtractor()
    serializers.load_model(args.subtractor_model, subtractor)
    return subtractor


def prepare_models(args):
    discriminator = Discriminator()
    generator = Generator()
    posterior = Posterior()
    transition = Transition()
    return discriminator, generator, posterior, transition


def show_dataset(dataset_dir):
    dataset = prepare_dataset(dataset_dir)
    print('dataset len: ', len(dataset))
    image1 = dataset[0][0]
    image2 = dataset[0][1]
    image3 = dataset[1][0]
    image4 = dataset[1][1]
    viewer.show_images(images=[image1, image3], title='current',
                       comparisons=[image2, image4], comparison_title='next', is_gray=True)


def load_images(images_dir):
    print('loading images from: ', images_dir)
    images = []
    files = os.listdir(images_dir)
    files.sort()
    for file_name in files:
        path = os.path.join(images_dir, file_name)
        if os.path.isfile(path) and '.jpg' in path:
            image = cv2.imread(path)
            image = cv2.resize(image, dsize=(64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = converter.hwc2chw(image) / 255.0
            image = image.astype(np.float32)
            assert np.all(0.0 <= image) and np.all(image <= 1.0)
            images.append(image)
    return images


def prepare_dataset(dataset_dir):
    current_images = []
    next_images = []
    if os.path.isdir(dataset_dir):
        images = load_images(dataset_dir)
        current_images.extend(images[0:-2])
        next_images.extend(images[1:-1])
    for file_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, file_name)
        if os.path.isdir(path):
            print('sub dir: ', file_name)
            images = load_images(path)
            current_images.extend(images[0:-2])
            next_images.extend(images[1:-1])
    return tuple_dataset.TupleDataset(current_images, next_images)


def prepare_iterator(args):
    dataset = prepare_dataset(args.dataset_dir)
    iterator = iterators.SerialIterator(
        dataset=dataset, batch_size=args.batch_size)
    return iterator


def prepare_updater(iterator, optimizers, subtractor, args):
    return GANUpdater(iterator=iterator, optimizer=optimizers, subtractor=subtractor, device=args.gpu)


def prepare_optimizers(discriminator, generator, posterior, transition, args):
    d_optimizer = setup_adam_optimizer(
        discriminator, lr=args.d_learning_rate, beta1=0.5, beta2=0.999)
    g_optimizer = setup_adam_optimizer(
        generator, lr=args.g_learning_rate, beta1=0.5, beta2=0.999)
    p_optimizer = setup_adam_optimizer(
        posterior, lr=args.g_learning_rate, beta1=0.5, beta2=0.999)
    t_optimizer = setup_adam_optimizer(
        transition, lr=args.g_learning_rate, beta1=0.5, beta2=0.999)

    return {'d_optimizer': d_optimizer, 'g_optimizer': g_optimizer, 'p_optimizer': p_optimizer, 't_optimizer': t_optimizer}


@training.make_extension(trigger=(10, 'epoch'))
def save_sample_image(trainer):
    updater = trainer.updater
    g_optimizer = updater.get_optimizer('g_optimizer')
    t_optimizer = updater.get_optimizer('t_optimizer')

    generator = g_optimizer.target
    transition = t_optimizer.target

    image_generator = ImageGenerator(generator, transition)

    image_current, image_next = image_generator(generator.device.device.id)

    image_current.to_cpu()
    image_next.to_cpu()

    iterator = trainer.updater.get_iterator('main')
    filename_current = 'intermediate/epoch-{}-current.jpg'.format(
        iterator.epoch)
    filename_next = 'intermediate/epoch-{}-next.jpg'.format(iterator.epoch)

    print('image array: ', image_current.data[0][0][0])
    pyplot.imsave(filename_current, image_current.data[0][0], cmap='gray')
    pyplot.imsave(filename_next, image_next.data[0][0], cmap='gray')


def prepare_trainer(discriminator, generator, posterior, transition,
                    updater, outdir, args):

    evaluation_trigger = (args.evaluation_interval, 'epoch')
    stop_trigger = (args.max_epochs, 'epoch')

    trainer = training.Trainer(updater,
                               stop_trigger=stop_trigger,
                               out=outdir)
    trainer.extend(training.extensions.LogReport(log_name="training_results",
                                                 trigger=evaluation_trigger))
    trainer.extend(training.extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'),
        trigger=evaluation_trigger)
    trainer.extend(training.extensions.snapshot_object(
        discriminator,
        filename='discriminator_epoch-{.updater.epoch}'),
        trigger=evaluation_trigger)
    trainer.extend(training.extensions.snapshot_object(
        generator,
        filename='generator_epoch-{.updater.epoch}'),
        trigger=evaluation_trigger)
    trainer.extend(training.extensions.snapshot_object(
        posterior,
        filename='posterior_epoch-{.updater.epoch}'),
        trigger=evaluation_trigger)
    trainer.extend(training.extensions.snapshot_object(
        transition,
        filename='transision_epoch-{.updater.epoch}'),
        trigger=evaluation_trigger)
    trainer.extend(save_sample_image)

    entries = ['epoch', 'd_loss', 'gpt_loss', 'elapsed_time']
    trainer.extend(training.extensions.PrintReport(entries=entries))
    trainer.extend(training.extensions.dump_graph('d_loss', 'gpt_loss'))

    return trainer


def run_training_loop(trainer, args):
    print('training started.')
    trainer.run()

def show_sample_subtraction(model, args):
    image = cv2.imread(args.test_subtraction_image)
    image = cv2.resize(image, dsize=(64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = converter.hwc2chw(image) / 255.0
    image = image.astype(np.float32)
    image = np.reshape(image, newshape=((1, ) + image.shape))
    print('image shape: ', image.shape)

    grayscale = model(image)
    grayscale = np.reshape(grayscale.data, newshape=(64, 64))
    print('grayscale shape: ', grayscale.shape)

    viewer.show_image(grayscale, is_gray=True)


def start_training(args):
    discriminator, generator, posterior, transition = prepare_models(args)
    if not args.gpu < 0:
        discriminator.to_gpu()
        generator.to_gpu()
        posterior.to_gpu()
        transition.to_gpu()

    subtractor = prepare_subtractor(args)
    if args.test_subtraction_image:
        show_sample_subtraction(subtractor, args)
    if not args.gpu < 0:
        subtractor.to_gpu()

    iterator = prepare_iterator(args)

    optimizers = prepare_optimizers(
        discriminator, generator, posterior, transition, args)

    updater = prepare_updater(iterator, optimizers, subtractor, args)

    outdir = files.prepare_output_dir(
        base_dir=args.outdir, args=args, time_format='')

    trainer = prepare_trainer(discriminator, generator, posterior, transition,
                              updater, outdir, args)

    run_training_loop(trainer, args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, required=True)

    # Training paramters
    parser.add_argument('--evaluation-interval', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--d-learning-rate', type=float, default=2*1e-4)
    parser.add_argument('--g-learning-rate', type=float, default=2*1e-4)

    # gpu setting
    parser.add_argument('--gpu', type=int, default=0)

    # data saving options
    parser.add_argument('--outdir', type=str, default='.')

    # model paths
    parser.add_argument('--subtractor-model', type=str, default='subtractor.model')

    # debug options 
    parser.add_argument('--test-subtraction-image', type=str, default=None)

    args = parser.parse_args()

    # show_dataset(args.dataset_dir)
    start_training(args)


if __name__ == "__main__":
    main()
