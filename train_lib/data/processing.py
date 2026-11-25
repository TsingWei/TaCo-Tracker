"""Prepare one template/search pair with the tracking training recipe."""

from .bounding_box_utils import flip_box, jitter_box
from .processing_utils import crop_window, sample_target, transform_image_to_crop
from .tensor import TensorDict
from .transforms import Normalize, crop_augmentation, grayscale, joint_augmentation


class PairProcessing:
    def __init__(self, config, training):
        self.config = config
        self.training = training
        self.normalize = Normalize(config["mean"], config["std"])

    def __call__(self, data):
        gray, joint_flip = joint_augmentation(self.training)
        result = TensorDict()
        for name in ("template", "search"):
            image, box = data[name + "_images"], data[name + "_anno"]
            crop = self.config[name]
            if gray:
                image = grayscale(image)
            if joint_flip:
                image = image[:, ::-1]
                box = flip_box(box, image.shape[1] - 1)
            extraction = jitter_box(box, crop["scale_jitter"], crop["center_jitter"])
            window = crop_window(extraction, crop["factor"])
            box = transform_image_to_crop(box, extraction, window[2], crop["size"])
            image = sample_target(image, window, crop["size"])
            crop_flip, brightness = crop_augmentation(self.training)
            if crop_flip:
                image = image[:, ::-1]
                box = flip_box(box)
            result[name + "_images"] = self.normalize(image, brightness)
            result[name + "_anno"] = box
        return result
