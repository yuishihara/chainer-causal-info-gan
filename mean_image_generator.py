import argparse

import os

import numpy as np

import cv2

from researchutils.image import viewer
from researchutils.image import converter


def load_images(images_dir):
    print('loading images from: ', images_dir)
    images = []
    files = os.listdir(images_dir)
    files.sort()
    for file_name in files:
        path = os.path.join(images_dir, file_name)
        if os.path.isfile(path) and '.jpg' in path:
            image = cv2.imread(path)
            image = np.asarray(image, dtype=np.uint8)
            images.append(image)
    return images


def generate_mean_image(args):
    dataset_dir = args.dataset_dir

    mean_image = None
    total_image_num = 0
    if os.path.isdir(dataset_dir):
        images = load_images(dataset_dir)
        mean_image = np.sum(images, axis=0)
        total_image_num += len(images)
    for file_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, file_name)
        if os.path.isdir(path):
            print('sub dir: ', file_name)
            images = load_images(path)
            if mean_image is not None:
                mean_image += np.sum(images, axis=0)
                total_image_num += len(images)
            else:
                mean_image = np.sum(images, axis=0)
                total_image_num += len(images)
    mean_image = mean_image / total_image_num
    mean_image = np.asarray(mean_image, dtype=np.uint8)

    viewer.show_image(cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB))

    cv2.imwrite('mean.jpg', mean_image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, required=True)

    args = parser.parse_args()

    generate_mean_image(args)


if __name__ == "__main__":
    main()
