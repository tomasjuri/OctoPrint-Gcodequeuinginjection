"""
ONNX patchwise CNN inference for render-reality matching.

Extracts patches from non-black regions of the render, runs them through
the Siamese CNN, and makes a pass/fail decision based on patch scores.
"""

import time
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_patch(patch_rgb, cnn_input_size):
    """Resize patch to cnn_input_size and apply ImageNet normalization.

    Args:
        patch_rgb: PIL Image (RGB) of arbitrary size.
        cnn_input_size: Target size (e.g. 224).

    Returns:
        numpy array of shape (3, cnn_input_size, cnn_input_size), float32.
    """
    resized = patch_rgb.resize((cnn_input_size, cnn_input_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # (3, H, W)


def extract_patches(capture_img, render_img, patch_size, overlap, cnn_input_size,
                    min_nonblack_ratio=0.05):
    """Extract aligned patches from capture and render images.

    Only keeps patches where the render has enough non-black pixels.

    Args:
        capture_img: PIL Image (RGB), full resolution.
        render_img: PIL Image (RGB), full resolution.
        patch_size: Crop size in pixels (e.g. 448).
        overlap: Overlap fraction (e.g. 0.5 for 50%).
        cnn_input_size: Resize target for CNN (e.g. 224).
        min_nonblack_ratio: Minimum fraction of non-black pixels in render patch.

    Returns:
        Tuple of (source_batch, render_batch, patch_locations):
          - source_batch: np.array (N, 3, H, W), float32
          - render_batch: np.array (N, 3, H, W), float32
          - patch_locations: list of (x, y, patch_size, patch_size) tuples
    """
    w, h = capture_img.size
    stride = int(patch_size * (1.0 - overlap))
    if stride < 1:
        stride = 1

    render_arr = np.array(render_img)  # (H, W, 3)

    source_patches = []
    render_patches = []
    locations = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Check non-black pixels in render patch
            render_crop = render_arr[y:y + patch_size, x:x + patch_size]
            nonblack = np.any(render_crop > 10, axis=2)  # any channel > 10
            ratio = nonblack.sum() / nonblack.size

            if ratio < min_nonblack_ratio:
                continue

            # Crop and normalize both
            src_crop = capture_img.crop((x, y, x + patch_size, y + patch_size))
            rnd_crop = render_img.crop((x, y, x + patch_size, y + patch_size))

            source_patches.append(normalize_patch(src_crop, cnn_input_size))
            render_patches.append(normalize_patch(rnd_crop, cnn_input_size))
            locations.append((x, y, patch_size, patch_size))

    if not source_patches:
        logger.warning("No valid patches found (render may be all black)")
        return np.empty((0, 3, cnn_input_size, cnn_input_size), dtype=np.float32), \
               np.empty((0, 3, cnn_input_size, cnn_input_size), dtype=np.float32), []

    source_batch = np.stack(source_patches).astype(np.float32)
    render_batch = np.stack(render_patches).astype(np.float32)

    logger.info("Extracted %d patches (patch_size=%d, stride=%d, image=%dx%d)",
                len(locations), patch_size, stride, w, h)
    return source_batch, render_batch, locations


def extract_center_patches(capture_img, render_img, patch_size, cnn_input_size,
                           n_patches=4, min_nonblack_ratio=0.05):
    """Extract a small number of patches from the center of the rendered object.

    Finds the centroid of non-black pixels in the render and places patches
    around it in a 2x2 (or smaller) grid with no overlap.

    Args:
        capture_img: PIL Image (RGB), full resolution.
        render_img: PIL Image (RGB), full resolution.
        patch_size: Crop size in pixels (e.g. 448).
        cnn_input_size: Resize target for CNN (e.g. 224).
        n_patches: Max number of center patches to return.
        min_nonblack_ratio: Minimum non-black pixel fraction per patch.

    Returns:
        Same tuple format as extract_patches.
    """
    w, h = capture_img.size
    render_arr = np.array(render_img)  # (H, W, 3)
    nonblack_mask = np.any(render_arr > 10, axis=2)

    if nonblack_mask.sum() == 0:
        logger.warning("Render is all black, no center patches")
        empty = np.empty((0, 3, cnn_input_size, cnn_input_size), dtype=np.float32)
        return empty, empty, []

    # Find centroid of non-black pixels
    ys, xs = np.where(nonblack_mask)
    cx, cy = int(xs.mean()), int(ys.mean())

    # Build candidate grid around centroid (2x2 touching patches)
    half = patch_size // 2
    offsets = [(-half, -half), (0, -half), (-half, 0), (0, 0)]

    source_patches = []
    render_patches = []
    locations = []

    for dx, dy in offsets:
        x = cx + dx
        y = cy + dy
        # Clamp to image bounds
        x = max(0, min(x, w - patch_size))
        y = max(0, min(y, h - patch_size))

        crop = render_arr[y:y + patch_size, x:x + patch_size]
        nb = np.any(crop > 10, axis=2)
        if nb.sum() / nb.size < min_nonblack_ratio:
            continue

        src_crop = capture_img.crop((x, y, x + patch_size, y + patch_size))
        rnd_crop = render_img.crop((x, y, x + patch_size, y + patch_size))
        source_patches.append(normalize_patch(src_crop, cnn_input_size))
        render_patches.append(normalize_patch(rnd_crop, cnn_input_size))
        locations.append((x, y, patch_size, patch_size))

        if len(locations) >= n_patches:
            break

    if not source_patches:
        empty = np.empty((0, 3, cnn_input_size, cnn_input_size), dtype=np.float32)
        return empty, empty, []

    logger.info("Extracted %d center patches around (%d, %d)", len(locations), cx, cy)
    return (np.stack(source_patches).astype(np.float32),
            np.stack(render_patches).astype(np.float32),
            locations)


class InferenceSession:
    """Thin wrapper around ONNX Runtime for the Siamese CNN."""

    def __init__(self, onnx_model_path):
        import onnxruntime as ort

        logger.info("Loading ONNX model: %s", onnx_model_path)
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        logger.info("ONNX model loaded. Inputs: %s, Outputs: %s", inputs, outputs)

    def run(self, source_batch, render_batch):
        """Run inference on a batch of patch pairs.

        Args:
            source_batch: np.array (N, 3, H, W), float32.
            render_batch: np.array (N, 3, H, W), float32.

        Returns:
            np.array of shape (N,) with scores in [0, 1].
        """
        t0 = time.time()
        outputs = self.session.run(None, {
            "source": source_batch,
            "render": render_batch,
        })
        scores = outputs[0].flatten()
        elapsed = time.time() - t0
        logger.info("Inference: %d patches in %.2fs (%.1f patches/s)",
                     len(scores), elapsed, len(scores) / max(elapsed, 1e-6))
        return scores

    @staticmethod
    def decide(scores, ratio_threshold, score_threshold):
        """Make pass/fail decision from patch scores.

        Args:
            scores: np.array of per-patch scores.
            ratio_threshold: Fraction of patches that must pass (e.g. 0.9).
            score_threshold: Per-patch pass threshold (e.g. 0.5).

        Returns:
            Tuple of (passed: bool, pass_ratio: float, stats: dict).
        """
        if len(scores) == 0:
            logger.warning("No patches to decide on, defaulting to PASS")
            return True, 1.0, {"n_patches": 0}

        passing = (scores >= score_threshold).sum()
        ratio = passing / len(scores)
        passed = ratio >= ratio_threshold

        stats = {
            "n_patches": len(scores),
            "n_passing": int(passing),
            "pass_ratio": float(ratio),
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
        }

        verdict = "PASS" if passed else "FAIL"
        logger.info("Decision: %s -- %.1f%% patches passed (threshold: %.1f%%), "
                     "mean=%.3f, min=%.3f, max=%.3f",
                     verdict, ratio * 100, ratio_threshold * 100,
                     stats["mean_score"], stats["min_score"], stats["max_score"])
        return passed, ratio, stats
