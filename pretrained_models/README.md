# ImageNet backbone checkpoints

Save the ImageNet classification checkpoint for your variant to the path below. The [training configs](../configs/) use these paths by default. Run training from the repository root, or point `train.py --pretrained` at the checkpoint explicitly. Training loads backbone stages 1–3 automatically (`drop_s4=True`).

| Variant | File in this directory | SHA-256 |
|---|---|---|
| TaCo-T | `lowformer_E1.pt` | `853c8083e58122d524c9c5af41386469e40aac98402236667fc5e7e41cc1efee` |
| TaCo-S | `repvit_m1_1_distill_300e.pth` | `1b364220241273d54547ef6632de7f892f71009c300f8aa4236e107974574570` |
| TaCo-B | `repvit_m1_5_distill_300e.pth` | `b434320aed41372a9e061b602c35286d2022724d76384137eb3133eed9573433` |

## LowFormer-E1

Open **All Checkpoints** in the official LowFormer model zoo and download the E1 ImageNet checkpoint at `.exp/cls/imagenet/E1/checkpoint/evalmodel.pt`. If you download the entire archive, extract that file from it. Save it as `pretrained_models/lowformer_E1.pt`. [LowFormer model zoo](https://github.com/altair199797/LowFormer#model-zoo)

## RepViT-M1.1 and RepViT-M1.5

Run the matching download command from the repository root. [RepViT v1.0 release](https://github.com/THU-MIG/RepViT/releases/tag/v1.0)

```bash
curl -fL https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_distill_300e.pth \
  -o pretrained_models/repvit_m1_1_distill_300e.pth
curl -fL https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_300e.pth \
  -o pretrained_models/repvit_m1_5_distill_300e.pth
```

Use `shasum -a 256 pretrained_models/<filename>` (or `sha256sum` on Linux) and compare the output to the table before training. For tracking inference, use the [trained TaCo checkpoints](../weights/README.md).
