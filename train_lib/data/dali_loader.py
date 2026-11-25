"""Sample encoded frames on CPU workers and preprocess their pixels with DALI."""

from pathlib import Path

import numpy as np
import torch

from .bounding_box_utils import flip_box, jitter_box
from .dali_pipeline import INPUTS, build_dali_pipeline
from .processing_utils import affine_matrix, crop_window, transform_image_to_crop
from .tensor import TensorDict
from .transforms import crop_augmentation, joint_augmentation


def encoded_image(path):
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise OSError(f"Empty encoded frame: {path}")
    return encoded


class DaliProcessing:
    def __init__(self, config, training):
        self.config = config
        self.training = training

    def __call__(self, data):
        gray, joint_flip = joint_augmentation(self.training)
        record = {"gray": np.array([gray], dtype=np.float32)}
        for name, suffix in (("template", "t"), ("search", "s")):
            crop = self.config[name]
            box = data[name + "_anno"]
            extraction = jitter_box(box, crop["scale_jitter"], crop["center_jitter"])
            window = crop_window(extraction, crop["factor"])
            box = transform_image_to_crop(box, extraction, window[2], crop["size"]).clamp_min(0)
            crop_flip, brightness = crop_augmentation(self.training)
            flip = joint_flip ^ crop_flip
            record[f"jpeg_{suffix}"] = data[name + "_images"]
            record[f"matrix_{suffix}"] = affine_matrix(window, crop["size"], flip)
            record[f"bright_{suffix}"] = np.array([brightness], dtype=np.float32)
            record[name + "_anno"] = flip_box(box) if flip else box
        return record


def collate_dali(batch):
    inputs = {key: [np.asarray(sample[key]) for sample in batch] for key in INPUTS}
    return inputs, torch.stack([sample["template_anno"] for sample in batch]), torch.stack(
        [sample["search_anno"] for sample in batch]
    )


class DaliLoader:
    def __init__(self, loader, batch_size, device, data, seed):
        self.loader = loader
        self.batch_size = batch_size
        self.sampler = loader.sampler
        self.generator = loader.generator
        self.pipeline = build_dali_pipeline(batch_size, device.index, data, seed)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for inputs, template_box, search_box in self.loader:
            count = len(template_box)
            # Keep the pipeline batch size fixed while preserving every validation pair.
            padding = self.batch_size - count
            for name, values in inputs.items():
                self.pipeline.feed_input(name, values + values[-1:] * padding)
            template, search = self.pipeline.run()
            # The DALI pipeline can recycle its output buffers on the next iteration.
            yield TensorDict(
                template_images=torch.from_dlpack(template.as_tensor())[:count].clone(),
                search_images=torch.from_dlpack(search.as_tensor())[:count].clone(),
                template_anno=template_box,
                search_anno=search_box,
            )
