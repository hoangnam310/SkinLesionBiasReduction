# Bias and Fairness Metrics Implementation Plan

This plan details how we will compute the specific metrics requested by your PI (Top-1 Accuracy, Macro AUPRC, Macro AUROC, and per-class metrics broken down by Fitzpatrick type) and save them to a `logs` directory.

## Goal
To implement a robust evaluation system that accurately measures classification performance across different skin tone subgroups to detect and quantify bias.

## User Review Required

- Do you want to generate the metrics automatically at the very end of your existing `train_baseline_efficientnet.py` script, or would you prefer a standalone script (e.g., `src/evaluate.py`) that you can run on any saved checkpoint? (I propose modifying `src/baseline_metrics.py` to have the logic, updating the train script to use it, and creating a standalone `src/evaluate.py` for flexibility).
- Is a JSON output inside a `logs` directory sufficient for your reporting, or would you also like a CSV/TXT format? (I propose JSON as it handles nested metrics perfectly, along with printing a clean text summary to the console).

## Proposed Changes

### 1. Update `src/baseline_metrics.py`

#### [MODIFY] `src/baseline_metrics.py`
- Modify `collect_predictions` to return four arrays instead of two: `y_true`, `y_pred`, `y_prob` (the softmax probabilities required for AUROC and AUPRC), and `skin_tones`.
- Create a new function `compute_bias_metrics(y_true, y_pred, y_prob, skin_tones)`:
  - Compute global Top-1 Accuracy.
  - Compute global Macro AUROC and Macro AUPRC (using `sklearn.metrics.roc_auc_score` and `average_precision_score`).
  - Group predictions by Fitzpatrick type (1-6).
  - Inside each group, compute accuracy, and per-class AUROC / AUPRC for the 3 lesion types. Handle cases where a specific class might be missing from a subgroup (which causes AUROC calculation to fail).

### 2. Create Evaluation Script

#### [NEW] `src/evaluate.py`
- Create a standalone evaluation script that:
  - Loads a trained PyTorch model from a given `--checkpoint`.
  - Loads the validation/test dataset using `get_dataloader`.
  - Runs inference using `collect_predictions`.
  - Computes the bias metrics using `compute_bias_metrics`.
  - Saves a nicely formatted `evaluation_metrics.json` file inside a `logs/` directory (created automatically if it doesn't exist).
  - Prints a human-readable summary to the console for quick inspection.

### 3. Update Training Script (Optional but Recommended)

#### [MODIFY] `src/train_baseline_efficientnet.py`
- Update the final evaluation block (around line 332) to use the new `collect_predictions` signature.
- Call the new `compute_bias_metrics` alongside the existing metrics.
- Save everything into the existing `metrics.json` output so you don't lose this information at the end of a training run.

## Verification Plan

### Automated Tests
- Run the new `evaluate.py` script on the existing test baseline to ensure `sklearn` does not throw errors (especially the tricky multi-class AUROC and AUPRC calculations).
- Verify that the `logs/evaluation_metrics.json` is generated correctly and contains the F1-F6 breakdowns.

### Manual Verification
- Review the JSON output to ensure all 3 lesion classes and all 6 Fitzpatrick types are represented.
- Check that the numbers make logical sense (e.g. AUROC is between 0.5 and 1.0).
