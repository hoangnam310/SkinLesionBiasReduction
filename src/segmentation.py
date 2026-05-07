"""
Segmentation-aware preprocessing for skin lesion images.

Pipeline (from notebooks/auto_segmentation_test.ipynb and Pipeline_test_with_10_more_imgs.ipynb):
    1. Resize so the shorter side equals `seg_size`, then center-crop to seg_size x seg_size.
    2. Optionally produce a foreground mask via a SAM-family model (SAM2 / SAM3 / MedSAM3).
       If no model is available, fall back to an all-ones mask.
    3. Slide a square window (side = `crop_frac` * seg_size) over the image, score each
       window by either edge texture (Laplacian) or LAB color variation, and keep the
       highest-scoring window whose mask coverage >= `min_skin`.
    4. Downscale the chosen ROI crop to `out_size` for downstream training.

The mask backends are encapsulated as plain callables so callers do not import any
SAM library unless they actually use one.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


MaskFn = Callable[[np.ndarray], Optional[np.ndarray]]
"""image (H, W, 3) uint8 RGB -> boolean foreground mask (H, W) or None."""


# ---------------------------------------------------------------------------
# Step 1: resize + center crop
# ---------------------------------------------------------------------------

def resize_and_center_crop(image: np.ndarray, target: int) -> np.ndarray:
    H, W = image.shape[:2]
    if H == 0 or W == 0:
        raise ValueError(f"Empty image with shape {image.shape}")

    scale = target / min(H, W)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (new_w - target) // 2)
    y0 = max(0, (new_h - target) // 2)
    crop = resized[y0:y0 + target, x0:x0 + target]

    if crop.shape[0] != target or crop.shape[1] != target:
        crop = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
    return crop


# ---------------------------------------------------------------------------
# Step 3: mask-aware ROI selection
# ---------------------------------------------------------------------------

def _best_crop_with_mask(
    score_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    crop_frac: float,
    min_skin: float,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    H, W = image.shape[:2]
    side = max(32, min(int(min(H, W) * crop_frac), min(H, W)))

    score_smooth = cv2.GaussianBlur(score_map.astype(np.float32), (0, 0), sigmaX=7)
    win_score = cv2.boxFilter(score_smooth, ddepth=-1, ksize=(side, side), normalize=False)
    win_skin = cv2.boxFilter(mask.astype(np.float32), ddepth=-1, ksize=(side, side), normalize=True)

    allowed = win_skin >= min_skin
    if allowed.any():
        masked_score = win_score.copy()
        masked_score[~allowed] = -1e18
        y, x = np.unravel_index(np.argmax(masked_score), masked_score.shape)
    else:
        y, x = np.unravel_index(np.argmax(win_score), win_score.shape)

    x0 = int(np.clip(x - side // 2, 0, W - side))
    y0 = int(np.clip(y - side // 2, 0, H - side))
    crop = image[y0:y0 + side, x0:x0 + side]
    return crop, (x0, y0, side)


def masked_edge_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_frac: float = 0.6,
    min_skin: float = 0.85,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return _best_crop_with_mask(np.abs(lap), image, mask, crop_frac, min_skin)


def masked_color_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_frac: float = 0.6,
    min_skin: float = 0.85,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    blur = cv2.GaussianBlur(lab, (0, 0), sigmaX=7)
    diff = np.sqrt(((lab - blur) ** 2).sum(axis=2))
    return _best_crop_with_mask(diff, image, mask, crop_frac, min_skin)


# ---------------------------------------------------------------------------
# SAM backends: each loader returns a MaskFn closure
# ---------------------------------------------------------------------------

def load_sam2_mask_generator(
    checkpoint: str,
    config: str = "sam2_hiera_t.yaml",
    device: Optional[str] = None,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.90,
    min_mask_region_area: int = 3000,
    overlap_thresh: float = 0.5,
) -> MaskFn:
    """SAM2 automatic mask generator. Picks the largest mask, then unions in any
    smaller masks that overlap the largest one by >= overlap_thresh, finishing
    with a morphological close + open. Mirrors notebooks cell 50 / 74."""

    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = device or _autodetect_device()
    model = build_sam2(config, checkpoint, device=device)
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    close_kernel = np.ones((11, 11), np.uint8)
    open_kernel = np.ones((7, 7), np.uint8)

    def _mask_fn(image: np.ndarray) -> Optional[np.ndarray]:
        anns = generator.generate(image)
        if not anns:
            return None
        anns.sort(key=lambda a: a["area"], reverse=True)
        base = anns[0]["segmentation"].astype(bool)
        union = base.copy()
        base_area = float(base.sum()) + 1e-6
        for a in anns[1:10]:
            m = a["segmentation"].astype(bool)
            inter = float((m & base).sum())
            if inter / base_area >= overlap_thresh:
                union |= m
        u8 = (union.astype(np.uint8) * 255)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, close_kernel)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, open_kernel)
        return u8.astype(bool)

    return _mask_fn


def load_sam3_mask_generator(
    prompt: str = "skin lesion",
    device: Optional[str] = None,
    score_thresh: float = 0.3,
) -> MaskFn:
    """Vanilla SAM3 with a text prompt. Returns the union of all masks above
    `score_thresh`, or the single highest-scoring mask if none clears the bar."""

    import torch  # noqa: F401  (driver-side init for SAM3)
    from PIL import Image as PILImage
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = device or _autodetect_device()
    model = build_sam3_image_model()
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    processor = Sam3Processor(model)

    def _mask_fn(image: np.ndarray) -> Optional[np.ndarray]:
        pil = PILImage.fromarray(image)
        state = processor.set_image(pil)
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks = out.get("masks")
        scores = out.get("scores")
        if masks is None or len(masks) == 0:
            return None
        masks_np = _as_numpy(masks).astype(bool)
        scores_np = _as_numpy(scores) if scores is not None else None
        return _select_mask_union(masks_np, scores_np, score_thresh, image.shape[:2])

    return _mask_fn


def load_medsam3_mask_generator(
    config_path: str,
    weights_path: str,
    prompt: str = "skin lesion",
    resolution: int = 1008,
    detection_threshold: float = 0.3,
    device: Optional[str] = None,
) -> MaskFn:
    """MedSAM3 (SAM3 + LoRA, fine-tuned for medical imagery, repo Joey-S-Liu/MedSAM3).

    `config_path` and `weights_path` are the LoRA training config and `.pt` checkpoint
    produced by the MedSAM3 repo. The script `infer_sam.py` from that repo must be on
    PYTHONPATH (it is the file that defines `SAM3LoRAInference`)."""

    from infer_sam import SAM3LoRAInference  # MedSAM3 repo

    device = device or _autodetect_device()
    inferencer = SAM3LoRAInference(
        config_path=config_path,
        weights_path=weights_path,
        resolution=resolution,
        detection_threshold=detection_threshold,
        device=device,
    )

    def _mask_fn(image: np.ndarray) -> Optional[np.ndarray]:
        # SAM3LoRAInference.predict() reads from disk; use its lower-level path instead.
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        results = _medsam3_predict_pil(inferencer, pil, [prompt])
        result = results.get(0)
        if not result or result["num_detections"] == 0 or result["masks"] is None:
            return None
        masks_np = result["masks"].astype(bool)  # (n, H, W) at original PIL size
        scores_np = result["scores"]
        return _select_mask_union(masks_np, scores_np, detection_threshold, image.shape[:2])

    return _mask_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _autodetect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_mask_union(
    masks_np: np.ndarray,
    scores_np: Optional[np.ndarray],
    score_thresh: float,
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Resize each mask to target_shape and union the ones that clear the threshold.
    Falls back to the highest-scoring mask when no mask passes the threshold."""

    if masks_np.ndim == 2:
        masks_np = masks_np[None]

    Ht, Wt = target_shape
    masks_resized: List[np.ndarray] = []
    for m in masks_np:
        if m.shape != (Ht, Wt):
            m = cv2.resize(m.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks_resized.append(m)

    if scores_np is not None and len(scores_np) == len(masks_resized):
        keep = [i for i, s in enumerate(scores_np) if s >= score_thresh]
        if not keep:
            keep = [int(np.argmax(scores_np))]
    else:
        keep = list(range(len(masks_resized)))

    union = np.zeros(target_shape, dtype=bool)
    for i in keep:
        union |= masks_resized[i]
    return union if union.any() else None


def _medsam3_predict_pil(inferencer, pil_image, prompts: List[str]) -> dict:
    """Mirror of SAM3LoRAInference.predict() but accepts an in-memory PIL image
    instead of a path. Avoids round-tripping through disk for every dataset image."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import nms
    from sam3.train.data.collator import collate_fn_api
    from sam3.model.utils.misc import copy_data_to_device

    results: dict = {}
    for query_idx, prompt in enumerate(prompts):
        datapoint = inferencer.create_datapoint(pil_image, [prompt])
        datapoint = inferencer.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = copy_data_to_device(batch, inferencer.device, non_blocking=True)
        with torch.no_grad():
            outputs = inferencer.model(batch)

        last = outputs[-1]
        pred_logits = last["pred_logits"]
        pred_boxes = last["pred_boxes"]
        pred_masks = last.get("pred_masks", None)

        scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
        keep = scores > inferencer.detection_threshold
        if int(keep.sum()) == 0:
            results[query_idx] = {"num_detections": 0, "scores": None, "masks": None}
            continue

        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        ow, oh = pil_image.size
        x1 = (cx - w / 2) * ow
        y1 = (cy - h / 2) * oh
        x2 = (cx + w / 2) * ow
        y2 = (cy + h / 2) * oh
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        keep_nms = nms(boxes_xyxy, kept_scores, inferencer.nms_iou_threshold)
        kept_scores = kept_scores[keep_nms]

        masks_np = None
        if pred_masks is not None:
            masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5
            masks_resized = F.interpolate(
                masks_small.unsqueeze(0).float(),
                size=(oh, ow),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5
            masks_np = masks_resized.cpu().numpy()

        results[query_idx] = {
            "num_detections": int(len(keep_nms)),
            "scores": kept_scores.cpu().numpy(),
            "masks": masks_np,
        }
    return results


# ---------------------------------------------------------------------------
# Public top-level pipeline
# ---------------------------------------------------------------------------

def process_image(
    image: np.ndarray,
    mask_fn: Optional[MaskFn] = None,
    strategy: str = "color",
    seg_size: int = 256,
    out_size: int = 64,
    crop_frac: float = 0.6,
    min_skin: float = 0.85,
) -> np.ndarray:
    """Full segmentation-aware preprocessing pipeline.

    Args:
        image: uint8 RGB numpy array of any size.
        mask_fn: callable returning a foreground mask, or None to skip SAM and use
            an all-ones mask (no segmentation, ROI selection only).
        strategy: 'edge', 'color', or 'center'. 'center' skips ROI search and
            returns the resize+center-crop output directly.
        seg_size: resolution at which segmentation and ROI search happen.
        out_size: final output side length (downstream training resolution).
        crop_frac: ROI window side as a fraction of seg_size.
        min_skin: minimum mask coverage required for a candidate window.

    Returns:
        uint8 RGB numpy array of shape (out_size, out_size, 3).
    """
    if strategy not in {"edge", "color", "center"}:
        raise ValueError(f"strategy must be one of edge|color|center, got {strategy!r}")

    step1 = resize_and_center_crop(image, target=seg_size)

    if strategy == "center":
        return _resize_if_needed(step1, out_size)

    mask: Optional[np.ndarray] = None
    if mask_fn is not None:
        mask = mask_fn(step1)
    if mask is None:
        mask = np.ones(step1.shape[:2], dtype=bool)

    if strategy == "edge":
        crop, _ = masked_edge_crop(step1, mask, crop_frac=crop_frac, min_skin=min_skin)
    else:
        crop, _ = masked_color_crop(step1, mask, crop_frac=crop_frac, min_skin=min_skin)

    return _resize_if_needed(crop, out_size)


def _resize_if_needed(image: np.ndarray, target: int) -> np.ndarray:
    if image.shape[0] == target and image.shape[1] == target:
        return image
    return cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Picklable preprocess_fn for DataLoader workers
# ---------------------------------------------------------------------------

class NoSegPreprocess:
    """Picklable preprocessor for the no-segmentation path (no SAM model).

    Pass an instance of this class as `preprocess_fn` to SkinLesionDataset to apply
    the segmentation-aware pipeline at load time. Backed by `process_image` with
    `mask_fn=None`, so DataLoader workers can fork it safely. SAM-based preprocessing
    should be done offline via `src/preprocess_segmentation.py`."""

    def __init__(
        self,
        strategy: str = "color",
        seg_size: int = 256,
        out_size: int = 64,
        crop_frac: float = 0.6,
        min_skin: float = 0.85,
    ):
        self.strategy = strategy
        self.seg_size = seg_size
        self.out_size = out_size
        self.crop_frac = crop_frac
        self.min_skin = min_skin

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return process_image(
            image,
            mask_fn=None,
            strategy=self.strategy,
            seg_size=self.seg_size,
            out_size=self.out_size,
            crop_frac=self.crop_frac,
            min_skin=self.min_skin,
        )


if __name__ == "__main__":
    # Smoke test: synthetic image through the no-segmentation pipeline.
    rng = np.random.default_rng(0)
    fake = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    out = process_image(fake, mask_fn=None, strategy="color", seg_size=128, out_size=64)
    print("out:", out.shape, out.dtype, "range:", out.min(), out.max())
    pre = NoSegPreprocess(strategy="edge", seg_size=128, out_size=64)
    print("NoSegPreprocess:", pre(fake).shape)
