# StyleGAN-XL QAT + TurboQuant

Progressive Quantization-Aware Fine-Tuning of the pretrained StyleGAN-XL
ImageNet 512² generator, followed by TurboQuant (random Hadamard rotation +
INT4) for further compression. Optional discriminator-guided Langevin
refinement is stubbed for v2.

## Setup

1. Clone StyleGAN-XL into the repo root:
   `git clone https://github.com/autonomousvision/stylegan-xl`
2. `pip install -r requirements.txt`
3. Download the pretrained 512² ImageNet checkpoint into `checkpoints/`.
4. Build a dataset zip with StyleGAN-XL's `dataset_tool.py`.
5. `python scripts/sanity_check.py`

## Pipeline

| Stage                          | Script                          |
| ------------------------------ | ------------------------------- |
| Sanity check                   | `scripts/sanity_check.py`       |
| FP32 baseline metrics          | `scripts/compute_baseline.py`   |
| Calibrate observers            | `scripts/calibrate.py`          |
| Progressive QAT (~20k iters)   | `scripts/train_qat.py`          |
| TurboQuant (post-QAT INT4)     | `scripts/apply_turboquant.py`   |
| Evaluate all variants          | `scripts/evaluate_all.py`       |

## Methodology

1. **Progressive QAT scheduling** ([qat/progressive.py](qat/progressive.py))
   — low-resolution blocks are quantized first; high-res blocks join later.
   `toRGB` and the mapping network stay in FP16 throughout.
2. **TurboQuant** ([turboquant/](turboquant/)) — random Hadamard rotation
   plus per-channel INT4 quantization of synthesis-network weights.
3. **Langevin refinement** ([refinement/langevin.py](refinement/langevin.py))
   — skeleton only in v1.

## Memory Optimizations

- BF16 mixed-precision training (B200-friendly)
- Gradient checkpointing on each synthesis block
- 8-bit Adam (`bitsandbytes`)
- Frozen discriminator (no grads, no optimizer state)
- FP16 EMA shadow, updated every 4 steps
- Pretrained feature extractors kept in BF16 with no grads

## Hardware Target

Single NVIDIA B200 (192 GB VRAM). Tuned for ~$400 GPU-time budget on Elice.
