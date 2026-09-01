# DGL-UIE

**DGL-UIE: Degradation-Guided Lightweight Underwater Image Enhancement**

Official implementation for the manuscript:

> **Real-Time Lightweight Underwater Image Enhancement with Hierarchical Small–Large Model Collaboration for Marine Monitoring**

DGL-UIE is a lightweight underwater image enhancement network designed for real-time edge deployment. The model combines color–illumination correction, degradation-adaptive frequency restoration, structural-gradient enhancement, and lightweight encoder–decoder processing.

## Highlights

- **Degradation-guided restoration:** color–illumination, frequency-energy, and structural-gradient degradation are handled by dedicated lightweight components.
- **HVI-BCC prior:** brightness, contrast, and chromatic correction are decoupled in an HVI-like representation.
- **Phase-preserving frequency restoration:** the Fourier amplitude is adaptively modulated while phase information is preserved.
- **Structural-gradient enhancement:** Sobel-gradient priors are integrated into a lightweight learnable structural block.
- **Lightweight architecture:** depthwise-separable convolutions and compact encoder–decoder blocks are used throughout the network.
- **Edge-oriented evaluation:** the code includes paired and no-reference evaluation as well as parameter/FLOP profiling.

## Repository Structure

```text
DGL-UIE/
├── model.py
├── train.py
├── test.py
├── test_nr.py
├── requirements.txt
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   ├── loss_funcs.py
│   ├── HVI_transform.py
│   ├── CIDNet.py
│   ├── CIDNet_weight_LOLv2_bestSSIM.pth
│   └── uranker/
│       ├── URanker.yaml
│       ├── URanker_ckpt.pth
│       └── uranker_utils.py
├── checkpoints/
│   └── best_model.pth
└── reproducibility/
    └── <100-point time-series CSV and related scripts>
```

The exact location of checkpoints and reproducibility files can be changed, but the corresponding paths in the commands below should be updated accordingly.

## Requirements

The Python dependencies currently listed in `requirements.txt` are:

```text
timm
einops
tqdm
numpy
matplotlib
Pillow
opencv-python
triton
thop
scikit-learn
scipy
scikit-image
huggingface_hub
```

Install them with:

```bash
pip install -r requirements.txt
```

PyTorch is used throughout the project but is not pinned in the current `requirements.txt`. Install a CUDA-compatible PyTorch build appropriate for your system before running the code.

## Dataset Preparation

The current training/testing scripts expect the following relative directory layout:

```text
UnderWaterDataset/
├── UIEB/
│   ├── train/
│   ├── val/
│   └── test/
├── EUVP-Dark/
│   ├── train/
│   ├── val/
│   └── test/
├── EUVP-Scene/
│   ├── train/
│   ├── val/
│   └── test/
├── UFO-120/
│   ├── train/
│   ├── val/
│   └── test/
└── LSUI/
    ├── train/
    ├── val/
    └── test/
```

The paired-data scripts currently support:

- `UIEB`
- `EUVP-d`
- `EUVP-s`
- `UFO`
- `LSUI`

Please download the datasets from their original sources and organize them according to the expected directory structure. Dataset redistribution is not included in this repository.

### No-reference evaluation

`test_nr.py` currently exposes the dataset names:

- `CH60`
- `EUVP`
- `U45`

The dataset roots in the current script are machine-specific absolute paths. Replace them with local paths before running no-reference evaluation.

## Model

The main model is implemented in `model.py` and instantiated as:

```python
from model import myModel

model = myModel(
    in_channels=3,
    feature_channels=32,
    use_white_balance=True
)
```

The current implementation includes:

- `HVIBCCWhiteBalance`: HVI-based brightness/contrast/chroma correction with a Gray-World prior;
- `DAFFB`: degradation-adaptive Fourier frequency restoration with low/high-frequency routing and phase preservation;
- `SGFB`: Sobel-gradient-guided structural feature enhancement;
- `BasicBlock`: lightweight HIN-style residual block using depthwise-separable convolutions;
- a compact two-level encoder–decoder backbone.

## Training

`train.py` uses paired supervision and currently combines:

```text
L = L_charbonnier
  + 0.5 * L_hvi
  + 0.1 * L_ssim
  + 0.1 * L_perceptual
  + 0.1 * L_edge
```

The current script uses:

- optimizer: `AdamW`
- initial learning rate: `2e-3`
- weight decay: `1e-5`
- default scheduler: cosine annealing
- random seed: `7`
- input size: `256 x 256`

Example:

```bash
python train.py \
  --dataset UIEB \
  --epoch 100 \
  --train_batch_size 24 \
  --eval_batch_size 24 \
  --lr 2e-3 \
  --scheduler cosine
```

Available training datasets are:

```text
UIEB, LSUI, UFO, EUVP-s, EUVP-d
```

### Important training note

The current `train.py` contains a project-specific hard-coded warm-start checkpoint:

```python
checkpoint = torch.load(
    r'output\WWE-UIE\UIEB\20260827_001558\best_model.pth'
)
```

Before releasing or reproducing training on another machine, **remove this hard-coded loading step or replace it with a portable checkpoint argument**. The script already provides a `--resume` argument, which is the recommended interface for checkpoint-based continuation.

Training also requires:

```text
utils/CIDNet_weight_LOLv2_bestSSIM.pth
```

for the HVI-domain supervision used in the current loss implementation.

## Paired Evaluation

Use `test.py` for paired evaluation. The script reports PSNR and SSIM, saves restored images, and profiles model parameters and FLOPs with THOP.

Example:

```bash
python test.py \
  --ckpt checkpoints/best_model.pth \
  --dataset UIEB \
  --test_batch_size 1
```

For paper-style single-frame inference, use batch size `1`.

The output directory is created next to the supplied checkpoint and includes restored predictions and `result.txt`.

## No-Reference Evaluation

Use `test_nr.py` for no-reference evaluation:

```bash
python test_nr.py \
  --ckpt checkpoints/best_model.pth \
  --dataset EUVP
```

The current no-reference evaluator reports:

- NIQE
- UCIQE
- URanker

and also profiles model parameters and FLOPs.

Before running this script, update the dataset paths and make sure the following URanker files are available:

```text
utils/uranker/URanker.yaml
utils/uranker/URanker_ckpt.pth
```

## Complexity Profiling

Both testing scripts use THOP:

```python
from thop import profile, clever_format
```

with a dummy input of size:

```text
1 x 3 x 256 x 256
```

to report model parameter count and FLOPs.

## Reproducing the Paper Results

The manuscript reports the following headline results for the proposed method:

| Setting | Result |
|---|---:|
| Parameters | 0.82 M |
| FLOPs | 6.70 G |
| Jetson Orin NX inference | ~9.0 ms/frame |
| Jetson Orin NX throughput | ~111 FPS |
| EUVP PSNR / SSIM | 29.977 dB / 0.932 |
| UIEB PSNR / SSIM | 24.046 dB / 0.912 |
| Communication saving | 93.8% |
| Event recall | 97.6% |
| Regional anomaly F1 | 0.814 |

For strict reproduction of the real-time numbers, use the same inference protocol reported in the manuscript:

```text
Hardware: NVIDIA Jetson Orin NX
Input size: 256 x 256
Batch size: 1
Precision: FP16
```

A dedicated latency/FPS benchmark script should synchronize CUDA before and after the timed inference region and should exclude data loading and image saving from the measured model latency.

## 100-Point Spatio-Temporal Fusion Data

The repository accompanying the manuscript should include the 100-point time-series data used for the single-buoy spatio-temporal fusion experiment and the corresponding Table/Figure reproduction scripts.

Recommended organization:

```text
reproducibility/
├── 100_point_timeseries.csv
├── reproduce_table6.py
└── reproduce_fig8.py
```

If your repository uses different filenames, update this section to match the released files.

## Pretrained Weights

Place the released DGL-UIE checkpoint at, for example:

```text
checkpoints/best_model.pth
```

and evaluate it with:

```bash
python test.py --ckpt checkpoints/best_model.pth --dataset UIEB --test_batch_size 1
```

Replace this section with the final checkpoint filename or release link used in the public repository.

## Reproducibility Checklist

Before making the repository public, verify that:

- [ ] all hard-coded local paths have been removed;
- [ ] the public checkpoint reproduces the reported quantitative results;
- [ ] the Jetson benchmark script uses the same input size, batch size, and FP16 protocol as the manuscript;
- [ ] the exact software/hardware environment is documented;
- [ ] the 100-point CSV and Table/Figure reproduction scripts are included;
- [ ] dataset download instructions point to the original sources;
- [ ] random seeds and evaluation settings are fixed and documented;
- [ ] the model/repository name is consistently changed to `DGL-UIE`.

## Citation

If you find this repository useful, please cite the associated manuscript:

```bibtex
@article{dgluie,
  title   = {Real-Time Lightweight Underwater Image Enhancement with Hierarchical Small--Large Model Collaboration for Marine Monitoring},
  author  = {To be updated},
  journal = {Journal of Real-Time Image Processing},
  year    = {To be updated}
}
```

The citation information should be updated after publication.

## Acknowledgment

This repository uses several auxiliary components and evaluation utilities. Please cite the corresponding original works when using those components in your own research.
