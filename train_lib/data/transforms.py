"""TaCo's joint grayscale/flip and independent crop augmentations."""

import random

import cv2
import numpy as np
import torch


def joint_augmentation(training):
    return training and random.random() < 0.05, training and random.random() < 0.5


def crop_augmentation(training):
    flip = training and random.random() < 0.5
    brightness = float(np.random.uniform(0.8, 1.2)) if training else 1.0
    return flip, brightness


def grayscale(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image, brightness):
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = image.float().mul_(brightness / 255.0).clamp_(0, 1)
        return image.sub_(self.mean).div_(self.std)
