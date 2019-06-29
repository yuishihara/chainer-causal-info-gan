import dill
import argparse

import chainer
import chainer.functions as F

import cv2

import numpy as np

import torch
import torch.nn as nn

from models.subtractor import Subtractor
from models.classifier import Classifier

from researchutils.chainer.serializers import save_model
from researchutils.chainer.serializers import load_model
from researchutils.image import converter
from researchutils.image import viewer


# Copied from https://github.com/thanard/causal-infogan/blob/master/model.py
# and removed unused code
class FCN_mse(nn.Module):
    """
    Predict whether pixels are part of the object or the background.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = torch.tanh(self.conv1(x))
        c2 = torch.tanh(self.conv2(c1))
        score = (self.classifier(c2))  # size=(N, n_class, H, W)
        return score  # size=(N, n_class, x.H/1, x.W/1)


# Copied from https://github.com/thanard/causal-infogan/blob/master/model.py
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


# Copied from https://github.com/thanard/causal-infogan/blob/master/model.py
class TorchClassifier(nn.Module):
    """
    Classifier is trained to predict the score between two black/white rope images.
    The score is high if they are within a few steps apart, and low other wise.
    """

    def __init__(self):
        super(TorchClassifier, self).__init__()
        self.LeNet = nn.Sequential(
            # input size 2 x 64 x 64. Take 2 black and white images.
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4),
            Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        stacked = torch.cat([x1, x2], dim=1)
        return self.LeNet(stacked)


def copy_weight(torch_weight, chainer_weight):
    assert torch_weight.shape == chainer_weight.shape
    chainer_weight.data = torch_weight


def copy_conv_weight(torch_conv, chainer_conv):
    assert chainer_conv.__class__.__name__ == 'Convolution2D'
    copy_weight(torch_conv.weight.detach().numpy(), chainer_conv.W)
    if torch_conv.bias is not None:
        copy_weight(torch_conv.bias.detach().numpy(), chainer_conv.b)


def copy_batch_norm_weight(torch_batch_norm, chainer_batch_norm):
    assert chainer_batch_norm.__class__.__name__ == "BatchNormalization"
    copy_weight(torch_batch_norm.weight.detach().numpy(),
                chainer_batch_norm.gamma)
    copy_weight(torch_batch_norm.bias.detach().numpy(),
                chainer_batch_norm.beta)
    copy_weight(torch_batch_norm.running_mean.detach().numpy(),
                chainer_batch_norm.avg_mean)
    copy_weight(torch_batch_norm.running_var.detach().numpy(),
                chainer_batch_norm.avg_var)


def convert_to_grayscale(subtractor, filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, dsize=(64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = converter.hwc2chw(image) / 255.0
    image = image.astype(np.float32)
    image = np.reshape(image, newshape=((1, ) + image.shape))
    print('image shape: ', image.shape)

    grayscale = subtractor(image)
    return grayscale


def show_sample_subtraction(model, args):
    grayscale = convert_to_grayscale(model, args.sample_image)
    grayscale = np.reshape(grayscale.data, newshape=(64, 64))
    print('grayscale shape: ', grayscale.shape)

    viewer.show_image(grayscale, is_gray=True)


def transfer_subtractor_params(args):
    pytorch_model = FCN_mse()
    pytorch_model.load_state_dict(torch.load(args.torch_subtractor_model_path))

    chainer_model = Subtractor()

    copy_conv_weight(pytorch_model.conv1, chainer_model.conv1)
    copy_conv_weight(pytorch_model.conv2, chainer_model.conv2)
    copy_conv_weight(pytorch_model.classifier, chainer_model.classifier)

    if args.sample_image:
        show_sample_subtraction(chainer_model, args)
    save_model(args.chainer_subtractor_save_path, chainer_model)


def classify_image_with_pytorch_model(pytorch_model, image1, image2):
    result = pytorch_model.forward(image1, image2)
    print('pytorch result: ', result.data)


def classify_image_with_chainer_model(chainer_model, image1, image2):
    concat_image = F.concat((image1, image2),  axis=1)
    result = chainer_model(concat_image)
    print('chainer result: ', result.data)


def transfer_classifier_params(args):
    pytorch_model = TorchClassifier()
    with open(args.torch_classifier_model_path, 'rb') as f:
        weights = dill.load(f)
    state_dict = {k: torch.FloatTensor(v) for k, v in weights[0].items()}
    pytorch_model.load_state_dict(state_dict)
    for k in state_dict.keys():
        print('key: ', k)

    chainer_model = Classifier()

    copy_conv_weight(pytorch_model.LeNet[0], chainer_model.conv1)
    copy_conv_weight(pytorch_model.LeNet[2], chainer_model.conv2)
    copy_batch_norm_weight(pytorch_model.LeNet[3], chainer_model.batch_norm1)
    copy_conv_weight(pytorch_model.LeNet[5], chainer_model.conv3)
    copy_batch_norm_weight(pytorch_model.LeNet[6], chainer_model.batch_norm2)
    copy_conv_weight(pytorch_model.LeNet[8], chainer_model.conv4)
    copy_batch_norm_weight(pytorch_model.LeNet[9], chainer_model.batch_norm3)
    copy_conv_weight(pytorch_model.LeNet[11], chainer_model.conv5)

    if args.sample_image:
        subtractor = Subtractor()
        load_model(args.chainer_subtractor_model_path, subtractor)
        image1 = convert_to_grayscale(subtractor, args.sample_image).data
        image2 = np.zeros(shape=image1.shape, dtype=np.float32)
        print('image1 shape: ', image1.shape)
        print('image2 shape: ', image2.shape)

        classify_image_with_pytorch_model(pytorch_model, torch.Tensor(image1), torch.Tensor(image2))
        classify_image_with_chainer_model(chainer_model, chainer.Variable(image1), chainer.Variable(image2))

    save_model(args.chainer_classifier_save_path, chainer_model)


def transfer_pytorch_to_chainer(args):
    if args.torch_subtractor_model_path is not None:
        transfer_subtractor_params(args)
    if args.torch_classifier_model_path is not None:
        transfer_classifier_params(args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--torch-subtractor-model-path', type=str, default=None)
    parser.add_argument('--chainer-subtractor-save-path', type=str,
                        default='./subtractor.model')

    parser.add_argument('--torch-classifier-model-path', type=str, default=None)
    parser.add_argument('--chainer-classifier-save-path', type=str,
                        default='./classifier.model')

    parser.add_argument('--chainer-subtractor-model-path', type=str, default=None)

    parser.add_argument('--sample-image', type=str, default=None)

    args = parser.parse_args()

    transfer_pytorch_to_chainer(args)


if __name__ == "__main__":
    # Transfer pytorch parameters to chainer model
    main()
