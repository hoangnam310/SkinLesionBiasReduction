# Auto-segmentation integration plan

## Context

Goal: integrate the lesion-focused preprocessing pipeline that two teammates prototyped in
`notebooks/auto_segmentation_test.ipynb` and `notebooks/Pipeline_test_with_10_more_imgs.ipynb`
into the main training code, so both the cGAN trainer and the EfficientNetV2-B0 baseline
can train on lesion-centered crops instead of whatever framing came from the source images.

Pipeline (mirrors notebook cells 50, 74, 75, 83):

1. Resize the original image so the shorter side equals `seg_size`, then center-crop to
   `seg_size × seg_size`.
2. Produce a foreground (lesion) mask via a SAM-family model. If no model is available
   the pipeline falls back to an all-ones mask, which still gives the cv2-only ROI search
   below something to work with.
3. Slide a square window (`crop_frac × seg_size` per side) over the cropped image, score
   each window by either Laplacian texture or LAB color variation, and keep the
   highest-scoring window whose mask coverage ≥ `min_skin`.
4. Downscale the chosen ROI to `out_size` for training.

Backends to support:

- **cv2-only** — runs anywhere, no model dependency. Fast enough to run inside a DataLoader
  worker at load time.
- **SAM2** (facebookresearch/sam2) — what the teammates' notebooks used. Largest-region
  union approach with morphological cleanup.
- **SAM3** (facebookresearch/sam3) — text-prompted; we'd pass `"skin lesion"`.
- **MedSAM3** (Joey-S-Liu/MedSAM3) — SAM3 + LoRA, fine-tuned on medical imagery; the most
  promising of the three for our domain.

The user wants training to run at 64×64 for speed. I don't want this anymore. Segmentation, however, needs higher
resolution (SAM models cannot produce useful masks at 64×64), so the pipeline runs at
`seg_size=256` and downscales the final crop to `out_size=64`. Only the saved/served
image is at 64; intermediate work stays at 256.

## Critical files

- `src/segmentation.py` _(new)_ — pipeline functions + backend loaders + a picklable
  `CV2Preprocess` wrapper for use as `preprocess_fn` in DataLoader workers.
- `src/preprocess_segmentation.py` _(new)_ — CLI that reads `fitzpatrick17k_cleaned.csv`,
  preprocesses every image referenced, and writes the result to `--output_dir`. Resume-by-default
  by checking `dst.exists()`.
- `src/data_process.py` _(edit)_ — `SkinLesionDataset` gains an optional
  `preprocess_fn: Callable[[np.ndarray], np.ndarray]` applied before the torchvision transform.
  Existing call sites are unaffected because the parameter defaults to `None`.

## Approach

### `src/segmentation.py`

Public surface:

- `resize_and_center_crop(image, target)` — Step 1.
- `masked_edge_crop(image, mask, ...)`, `masked_color_crop(image, mask, ...)` — Step 3.
- `process_image(image, mask_fn=None, strategy='color', seg_size=256, out_size=64,
crop_frac=0.6, min_skin=0.85)` — full pipeline as one call.
- `CV2Preprocess(strategy, seg_size, out_size, crop_frac, min_skin)` — picklable callable
  that wraps `process_image(..., mask_fn=None, ...)`. Pass an instance as
  `SkinLesionDataset(preprocess_fn=...)`.
- Backend loaders, each returning a `MaskFn = Callable[[np.ndarray], Optional[np.ndarray]]`:
  - `load_sam2_mask_generator(checkpoint, config='sam2_hiera_t.yaml', ...)`
  - `load_sam3_mask_generator(prompt='skin lesion', ...)`
  - `load_medsam3_mask_generator(config_path, weights_path, prompt='skin lesion', ...)`

Backend abstraction note: SAM2 is automatic (no prompt → many masks → take the largest +
overlapping siblings + morphology). SAM3 / MedSAM3 are text-prompted (`"skin lesion"` →
masks + scores → union all masks above `score_thresh`, fall back to top-1). The
`MaskFn` callable hides this difference from the rest of the pipeline.

MedSAM3 specifics: the `infer_sam.py` script in that repo defines `SAM3LoRAInference`,
which expects a config YAML and a LoRA `.pt`. Its public `predict()` reads from disk;
we re-implement the same forward pass over an in-memory PIL image inside a private
`_medsam3_predict_pil()` helper to avoid disk round-trips per dataset image.

### `src/preprocess_segmentation.py`

CLI flags:

- `--backend {cv2,sam2,sam3,medsam3}` (default `cv2`)
- `--strategy {edge,color,center}` (default `color`)
- `--seg_size 256 --out_size 64 --crop_frac 0.6 --min_skin 0.85`
- Backend-specific: `--sam2_checkpoint`, `--sam2_config`, `--prompt`,
  `--medsam3_config`, `--medsam3_weights`, `--medsam3_resolution`, `--detection_threshold`.
- Always-useful: `--limit N` (smoke test), `--no_resume` (force reprocess), `--device`.

Reads `dataset/fitzpatrick17k_cleaned.csv`, drops rows with `fitzpatrick_scale == -1` to
mirror `SkinLesionDataset`, preprocesses every image whose source `.jpg` exists, and saves
to `<output_dir>/<md5hash>.jpg`. Resume defaults to true (skip if dst exists).

After it runs, train against the new dir without code changes:

```
python src/train_baseline_efficientnet.py --image_dir dataset/images_<backend>_<strategy> \
    --image_size 64
```

### `src/data_process.py`

One additive parameter on `SkinLesionDataset`:

```python
preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

In `__getitem__`, applied between `Image.open(...).convert("RGB")` and `self.transform(...)`:

```python
if self.preprocess_fn is not None:
    image = Image.fromarray(self.preprocess_fn(np.array(image)))
```

No call sites need updating; `train.py`, `train_baseline_efficientnet.py`, etc. all keep
working as-is.

## State as of plan-mode entry

These edits were made before plan mode kicked in:

- ✅ `src/segmentation.py` — created (smoke test `python src/segmentation.py` passes:
  cv2-only `process_image` and `CV2Preprocess` both produce `(64, 64, 3) uint8` outputs).
- ✅ `src/preprocess_segmentation.py` — created (untested; cv2 path is the natural first run).
- ✅ `src/data_process.py` — `preprocess_fn` parameter wired into `__init__` and
  `__getitem__`; default of `None` keeps existing trainers unchanged.

If you want to keep the work, no further edits are required to land the integration. If
you want a different shape (e.g. fold the loaders into separate files per backend, or
expose backend selection via a `--seg_backend` flag on the trainer instead of running the
preprocessor offline), say so before exiting plan mode.

## Verification

1. **Module smoke test** — already passed before plan mode:
   ```
   python src/segmentation.py
   # expects: out: (64, 64, 3) uint8 ...   CV2Preprocess: (64, 64, 3)
   ```
2. **Dataset smoke test** with `preprocess_fn`:
   ```python
   from data_process import SkinLesionDataset
   from segmentation import CV2Preprocess
   ds = SkinLesionDataset(
       csv_path="dataset/fitzpatrick17k_cleaned.csv",
       image_dir="dataset/images",
       preprocess_fn=CV2Preprocess(strategy="color", seg_size=256, out_size=224),
   )
   img, st, lt = ds[0]
   ```
   Should return a tensor without errors.
3. **Offline preprocessing — cv2 dry run, 50 images:**
   ```
   python src/preprocess_segmentation.py --backend cv2 --strategy color \
       --output_dir dataset/images_cv2_color --limit 50
   ```
4. **Train against the preprocessed dir:**
   ```
   python src/train_baseline_efficientnet.py \
       --image_dir dataset/images_cv2_color --image_size 64 \
       --epochs 1 --max_steps 5 --no_save
   ```
5. **SAM2 dry run** (only if you've cloned/built sam2 + downloaded a checkpoint):
   ```
   python src/preprocess_segmentation.py --backend sam2 \
       --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt \
       --output_dir dataset/images_sam2_color --limit 20
   ```
6. **MedSAM3 dry run** (requires the MedSAM3 repo on PYTHONPATH and its LoRA weights):
   ```
   PYTHONPATH=external/MedSAM3 python src/preprocess_segmentation.py --backend medsam3 \
       --medsam3_config external/MedSAM3/configs/full_lora_config.yaml \
       --medsam3_weights external/MedSAM3/outputs/sam3_lora_full/best_lora_weights.pt \
       --output_dir dataset/images_medsam3_color --limit 10
   ```

## Decisions to confirm before exiting plan mode

- Default training resolution stays 64×64 (matches the user's note on faster training).
- Default `seg_size` is 256. SAM3/MedSAM3 internally upsample to ~1008 anyway, but 256
  is the resolution at which we score / pick the ROI before downscaling. Tunable.
- Default `strategy` is `color` (LAB color-variation). The notebooks ran both `edge` and
  `color`; color tends to land on the lesion more reliably. The CLI accepts either.
- Default `min_skin` is `0.85` — same as the final notebook setting (cell 77).
- MedSAM3 needs the upstream repo as a Python dependency. The plan keeps it as a
  `PYTHONPATH=external/MedSAM3` adjunct rather than pulling it into the `gan` conda env,
  so the local Mac dev loop is unaffected.

## Out of scope for this plan

- Adding a `--seg_backend` flag directly to `train_baseline_efficientnet.py` or `train.py`.
  Preferred path is offline preprocessing → point `--image_dir` at the new dir. Keeps SAM
  out of the training process entirely.
- Bias-aware re-evaluation of the segmented vs. raw classifier. Belongs in a follow-up
  comparing per-FST AUROC before/after segmentation.
- The 1-hour-per-epoch / 35%-accuracy diagnostics from the previous turn — separate
  thread; segmentation is unrelated.

python src/train_baseline_efficientnet.py --csv_path dataset/fitzpatrick17k_cleaned.csv --image_dir dataset/images_sam2_color --image_size 64 --epochs 20 --batch_size 32 --lr 1e-4 --weight_decay 1e-4 --num_workers 1 --class_weights --freeze_backbone --unfreeze_epoch 3 --fine_tune_lr 1e-5 --output_dir outputs/efficientnet_baseline_sam2 --device mps
