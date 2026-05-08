# SkinLesionBiasReduction

Skin-lesion datasets overrepresent lighter skin tones (Fitzpatrick types I–IV), so
classifiers trained on them tend to make more errors on darker tones (FST V–VI).
This project trains a **conditional WGAN-GP** to synthesize realistic images for
the under-represented tone × lesion-type cells, upsamples those cells in the
training set, and measures whether an EfficientNetV2-B0 classifier becomes more
equitable across Fitzpatrick subgroups (narrower TPR gaps, AUROC parity, better
calibration on darker tones).

The pipeline is run end-to-end against two preprocessing variants of the
Fitzpatrick17k dataset:

| Variant            | Preprocessing                                                  |
|--------------------|----------------------------------------------------------------|
| `center`           | Resize + center crop only                                      |
| `center_sam2_edge` | Center crop + SAM2 skin mask + edge ROI (background suppressed)|

Each variant is trained and evaluated independently so the effect of segmentation
on both GAN quality and downstream bias can be compared directly.

## Approach

- **Generator**: conditional on (Fitzpatrick tone, lesion type). Upsample + Conv
  blocks; SAGAN-style self-attention at 32×32.
- **Critic**: conditional, LayerNorm (Gulrajani 2017), self-attention at 32×32.
- **Loss**: Wasserstein loss with gradient penalty (`λ_gp = 10`, `n_critic = 5`).
- **Augmentation**: DiffAugment (`color, translation, cutout`) applied to both
  real and fake critic inputs.
- **Resolution**: 64×64 (matches the downstream classifier; allows fast
  iteration and FID feedback every N epochs).
- **Synthetic budget**: skewed toward FST 5/6 benign and malignant cells; small
  budget for non-neoplastic.

## Repository layout

```
src/
  cgan.py                       Generator + Critic (conditional, attention, norm options)
  diffaugment.py                DiffAugment (color/translation/cutout)
  train.py                      Conditional GAN trainer (BCE)
  train_wgan.py                 Conditional WGAN-GP trainer (used by the pipeline)
  generate.py                   Sample synthetics for given (FST, lesion) cells
  fid.py                        Standalone FID against a real reference set
  utils.py                      compute_fid, dataset, logging helpers
  preprocess_segmentation.py    Center crop + (optional) SAM2 mask + edge ROI
  segmentation.py               SAM2 wrapper
  data_process.py               Splits / partition labels / CSV utilities
  clean_dataset.py              Filtering and consistency checks
  prepare_m2m_dataset.py        M2M-style training set assembly
  merge_synthetic.py            Add synthetics to partition=train only
  train_baseline_efficientnet.py  EfficientNetV2-B0 trainer
  baseline_metrics.py           Classification + per-FST bias metrics
  evaluate.py                   Run a checkpoint on the test split, write JSON
  metrics_report.py             Render the JSON as a markdown report
scripts/
  find_best_checkpoint.py       Pick lowest-FID checkpoint from training.log
run_preprocess.sh               Build images_center and images_center_sam2_edge
run_main.sh                     Train both cGANs (center + sam2_edge)
run_generate.sh <variant>       Generate FST 5/6 synthetics from best checkpoint
run_baseline.sh                 EfficientNet on real-only data (control)
run_baseline_upsampled.sh <variant>   EfficientNet on GAN-upsampled training set
run_pipeline.sh                 End-to-end overnight run of all of the above
```

## Setup

The project pins a CUDA 11.8 build of PyTorch (sm_89 / RTX 4090). For a GPU box:

```bash
conda create -n gan python=3.10 -y
conda activate gan
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

For CPU-only or Mac MPS, install `torch / torchvision / torchaudio` separately
and ignore the `+cu118` pins.

SAM2 weights for the segmentation variant:

```
checkpoints/sam2_hiera_tiny.pt
```

The dataset CSV is `dataset/fitzpatrick17k_c.csv`; raw images live in
`dataset/images/` and preprocessing produces `dataset/images_center/` and
`dataset/images_center_sam2_edge/`.

## Running the pipeline

End-to-end (preprocess → train GAN → generate → train classifier → evaluate),
both variants:

```bash
./run_preprocess.sh
./run_pipeline.sh
```

Or step by step for one variant (e.g. `center`):

```bash
./run_preprocess.sh                       # one-time, builds both image dirs
./run_main.sh                             # trains both cGANs (edit IMAGE_DIRS to skip one)
./run_generate.sh center                  # samples FST 5/6 cells from the best checkpoint
./run_baseline.sh                         # control: real-only classifier
./run_baseline_upsampled.sh center        # treatment: classifier on GAN-upsampled set
```

Each GAN run writes to `outputs/wgan_<variant>_<timestamp>/` with
`checkpoints/`, `logs/training.log`, and `samples/`. Use
`scripts/find_best_checkpoint.py <run_dir>` to pick the lowest-FID checkpoint.

## Metrics

**GAN (during training, logged to TensorBoard)**
- `FID` — Fréchet Inception Distance vs. the real subset of the same variant
  (`src/utils.py:434`, computed every `--fid_interval` epochs). Lower is better;
  the only quality metric used to pick checkpoints.
- `Metric/Wasserstein` — `E[critic(real)] − E[critic(fake)]`, the critic's
  estimate of the EM distance. Should trend toward 0.
- `Metric/GradientPenalty` — `(‖∇ critic(x̂)‖₂ − 1)²` on α-interpolated samples;
  enforces the 1-Lipschitz constraint. Should stay small and stable.
- `Loss/Critic`, `Loss/Generator`, plus `Score/D(x)` and `Score/D(G(z))` on the
  BCE cGAN path (`src/train.py`).

**Classifier (per-Fitzpatrick bias metrics, `src/baseline_metrics.py`)**
- Global: top-1 accuracy, macro AUROC, macro AUPRC.
- Per-FST: AUROC, AUPRC, accuracy, TPR.
- Per-class: precision, recall, F1, support.
- Rendered as a markdown table by `src/metrics_report.py`.

## Outputs

```
outputs/wgan_<variant>_<timestamp>/        cGAN run (checkpoints, logs, samples)
outputs/baseline_efficientnet_<variant>/   classifier checkpoint + metrics
generated_images/<variant>/                synthesized JPEGs + generated_metadata.csv
dataset/fitzpatrick17k_c_upsampled_<variant>.csv   train-only merge of real + synthetic
logs/evaluation_metrics.json               bias metrics from src/evaluate.py
report.md                                  rendered bias report
```

## Notes

- Synthetics are added to `partition=train` only; `val` and `test` stay real-only
  so the bias evaluation never sees generated images.
- The two variants share GAN architecture, hyperparameters, and seeds — the only
  difference is the preprocessing applied to the input images, so any difference
  in downstream bias is attributable to the SAM2 + edge ROI step.
- `run_baseline.sh` and `run_baseline_upsampled.sh` use identical hyperparameters
  and seed for a clean A/B comparison.
