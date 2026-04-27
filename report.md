Step-by-step recipe

1. Smoke-test the pipeline (≈10 seconds, no training)

Verifies the new flag wires up without burning a real run:

python src/train_baseline_efficientnet.py \
 --image_size 64 --epochs 1 --max_steps 2 --batch_size 16 \
 --no_pretrained --no_freeze_backbone --class_weights \
 --no_save --device cpu

You should see the Class weights (...) line print and one fast epoch finish. If that works, the change is live.

2. Train a real small-image baseline

This is the run you'll iterate on. Three changes from before: class weights on, backbone unfrozen at epoch 3, image size still 64
for fast feedback:

python src/train_baseline_efficientnet.py \
 --image_size 64 \
 --epochs 20 \
 --batch_size 64 \
 --lr 1e-4 \
 --class_weights \
 --freeze_backbone --unfreeze_epoch 3 --fine_tune_lr 1e-5 \
 --device mps

What each flag is doing for you:

- --class_weights — fixes the "predict non-neoplastic always" failure mode. This is the main change.
- --freeze_backbone (default on) + --unfreeze_epoch 3 — head trains for 3 epochs on top of frozen ImageNet features, then the
  backbone joins fine-tuning at a lower LR. Without unfreezing, the frozen ImageNet features alone cap discrimination — and that
  ceiling is part of why the prior run hit AUROC 0.63.
- --epochs 20 — enough to see the unfreeze take effect; bump to 30–40 once you're happy with the recipe.
- --batch_size 64 should fit on MPS at 64×64; drop to 32 if you OOM.

Watch for: train_loss should drop noticeably after unfreeze_epoch. val_acc will likely drop vs the old baseline (because the model
is no longer hiding behind the majority class) — that is the desired behavior. Track macro AUROC and per-FST AUROC in the next step
instead.

3. Evaluate + report

LATEST=$(ls -td outputs/baseline_efficientnet/*/ | head -1)
  python src/evaluate.py --checkpoint "${LATEST}checkpoint.pt"
python src/metrics_report.py

The report markdown is what you'll diff against the previous baseline to judge whether the changes helped.

4. What to look at, in order

1. Macro AUROC (global) — should rise from 0.63.
1. Macro F1 / balanced accuracy — these will move much more than top-1 accuracy.
1. Malignant AUROC by FST — this is the headline bias slice. Hopefully the FST 1 vs FST 5 gap shrinks.
1. Top-1 accuracy — expect it to drop a few points; ignore that. With class weights, the model stops free-riding on the majority
   class.

1. When the 64×64 run looks reasonable, scale up

Same flags, just bump --image_size. EfficientNetV2-B0 was pretrained at 192–224, so 224 will recover most of the pretrained-features
benefit:

python src/train_baseline_efficientnet.py \
 --image_size 224 \
 --epochs 30 --batch_size 32 \
 --lr 1e-4 \
 --class_weights \
 --freeze_backbone --unfreeze_epoch 3 --fine_tune_lr 1e-5 \
 --device mps

Each epoch will be ~10× slower than 64×64, so don't scale up until the 64×64 recipe is dialed in.

Optional next lever (if class weights + unfreeze still aren't enough)

If macro F1 is still weak after step 2, the next minimal change is a WeightedRandomSampler on the train loader. That's a slightly
bigger code change (DataLoader can't take both shuffle=True and a sampler) — happy to add it if you want, but try class weights
alone first.
