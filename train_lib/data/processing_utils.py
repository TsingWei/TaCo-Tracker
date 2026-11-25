"""Square crop geometry shared by the torch and DALI pipelines."""

import math

import cv2
import numpy as np
import torch


def crop_window(box, factor):
    x, y, w, h = box.tolist()
    size = math.ceil(math.sqrt(w * h) * factor)
    if size < 1:
        raise ValueError("Target box is too small to crop")
    return round(x + (w - size) / 2), round(y + (h - size) / 2), size


def sample_target(image, window, output_size):
    x, y, size = window
    height, width = image.shape[:2]
    left, top = max(0, x), max(0, y)
    # Match the reference crop's end-of-image convention.
    right, bottom = min(x + size, width - 1), min(y + size, height - 1)
    if right <= left or bottom <= top:
        raise ValueError("Crop falls outside the image")
    crop = image[top:bottom, left:right]
    crop = cv2.copyMakeBorder(crop, top - y, y + size - bottom,
                              left - x, x + size - right, cv2.BORDER_CONSTANT)
    return cv2.resize(crop, (output_size, output_size))


def transform_image_to_crop(box, extraction_box, crop_size, output_size):
    """Map xywh into normalized crop coordinates using the jittered center."""
    scale = output_size / crop_size
    center = box[:2] + 0.5 * box[2:]
    extraction_center = extraction_box[:2] + 0.5 * extraction_box[2:]
    crop_center = (output_size - 1) / 2 + (center - extraction_center) * scale
    size = box[2:] * scale
    return torch.cat((crop_center - 0.5 * size, size)) / output_size


def affine_matrix(window, output_size, flip):
    """Destination-to-source transform for DALI's affine crop."""
    x, y, size = window
    scale = size / output_size
    if flip:
        return np.array([[-scale, 0, x + (output_size - 0.5) * scale],
                         [0, scale, y + 0.5 * scale]], dtype=np.float32)
    return np.array([[scale, 0, x + 0.5 * scale],
                     [0, scale, y + 0.5 * scale]], dtype=np.float32)
