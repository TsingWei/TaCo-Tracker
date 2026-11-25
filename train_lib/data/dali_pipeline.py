"""Optional DALI decode, affine crop and normalization on CUDA."""

INPUTS = ("jpeg_t", "jpeg_s", "matrix_t", "matrix_s", "bright_t", "bright_s", "gray")


def dali_available():
    try:
        from nvidia.dali import fn, pipeline_def, types  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def build_dali_pipeline(batch_size, device_id, data, seed):
    from nvidia.dali import fn, pipeline_def, types

    mean = [255.0 * value for value in data["mean"]]
    std = [255.0 * value for value in data["std"]]

    def branch(encoded, matrix, brightness, gray, size):
        image = fn.decoders.image(
            encoded, device="mixed", output_type=types.RGB,
            hw_decoder_load=0.65, cache_batch_copy=True,
            device_memory_padding=16 * 1024 * 1024, host_memory_padding=16 * 1024 * 1024,
        )
        image = fn.warp_affine(
            image, matrix=matrix, size=[size, size], fill_value=0.0,
            interp_type=types.INTERP_LINEAR, inverse_map=True,
        )
        image = fn.saturation(image, saturation=1.0 - gray)
        image = fn.brightness_contrast(image, brightness=brightness, dtype=types.UINT8)
        return fn.crop_mirror_normalize(image, mean=mean, std=std, dtype=types.FLOAT,
                                        output_layout="CHW")

    @pipeline_def(batch_size=batch_size, device_id=device_id,
                  num_threads=2, prefetch_queue_depth=1, seed=seed)
    def graph():
        inputs = {
            key: fn.external_source(
                name=key, dtype=types.UINT8 if key.startswith("jpeg") else types.FLOAT
            ) for key in INPUTS
        }
        return (
            branch(inputs["jpeg_t"], inputs["matrix_t"], inputs["bright_t"], inputs["gray"],
                   data["template"]["size"]),
            branch(inputs["jpeg_s"], inputs["matrix_s"], inputs["bright_s"], inputs["gray"],
                   data["search"]["size"]),
        )

    pipe = graph()
    pipe.build()
    return pipe
