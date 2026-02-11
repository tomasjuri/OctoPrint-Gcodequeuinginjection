"""
Visualization and result saving for inference results.

Creates heatmap images from patch scores and saves all artifacts
(capture, render, heatmap, metadata JSON) to disk.
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def create_heatmap(image_size, patch_locations, scores):
    """Create a heatmap image from patch locations and scores.

    Args:
        image_size: Tuple of (width, height) for the output image.
        patch_locations: List of (x, y, w, h) tuples.
        scores: np.array of per-patch scores in [0, 1].

    Returns:
        PIL Image (RGB) with heatmap overlay.
    """
    w, h = image_size
    # Accumulate scores and counts per pixel for averaging
    score_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for (px, py, pw, ph), score in zip(patch_locations, scores):
        score_sum[py:py + ph, px:px + pw] += score
        count[py:py + ph, px:px + pw] += 1

    # Average where we have data
    mask = count > 0
    avg = np.zeros_like(score_sum)
    avg[mask] = score_sum[mask] / count[mask]

    # Map to color: red (0) -> yellow (0.5) -> green (1)
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap[mask, 0] = (np.clip(1.0 - avg[mask], 0, 1) * 255).astype(np.uint8)  # R
    heatmap[mask, 1] = (np.clip(avg[mask], 0, 1) * 255).astype(np.uint8)          # G
    heatmap[mask, 2] = 0                                                            # B

    # Draw patch grid outlines
    img = Image.fromarray(heatmap)
    draw = ImageDraw.Draw(img)
    for (px, py, pw, ph), score in zip(patch_locations, scores):
        color = (0, 255, 0) if score >= 0.5 else (255, 0, 0)
        draw.rectangle([px, py, px + pw - 1, py + ph - 1], outline=color, width=1)

    logger.info("Created heatmap (%dx%d) from %d patches", w, h, len(scores))
    return img


def save_results(capture_img, render_img, heatmap_img, metadata, save_folder,
                 run_dir=None):
    """Save all inference artifacts to disk.

    Args:
        capture_img: PIL Image (RGB) -- camera capture.
        render_img: PIL Image (RGB) -- rendered gcode.
        heatmap_img: PIL Image (RGB) -- score heatmap.
        metadata: dict with inference results and metadata.
        save_folder: Base folder (e.g. ~/data/inference_results).
        run_dir: If provided, use this directory directly (one folder per
                 print job).  Otherwise fall back to creating a new
                 timestamped subfolder.

    Returns:
        Path to the save directory.
    """
    if run_dir is not None:
        save_dir = run_dir
    else:
        save_folder = os.path.expanduser(save_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcode_name = metadata.get("gcode_name", "unknown")
        save_dir = os.path.join(save_folder, f"{timestamp}_{gcode_name}")

    os.makedirs(save_dir, exist_ok=True)
    layer_n = metadata.get("layer_n", 0)

    # Each layer gets its own subfolder
    layer_dir = os.path.join(save_dir, f"layer_{layer_n:04d}")
    os.makedirs(layer_dir, exist_ok=True)

    # Save images
    capture_path = os.path.join(layer_dir, "capture.jpg")
    capture_img.save(capture_path)
    logger.info("Saved capture: %s", capture_path)

    render_path = os.path.join(layer_dir, "render.png")
    render_img.save(render_path)
    logger.info("Saved render: %s", render_path)

    heatmap_path = os.path.join(layer_dir, "heatmap.png")
    heatmap_img.save(heatmap_path)
    logger.info("Saved heatmap: %s", heatmap_path)

    # Save heatmap-over-capture overlay
    if capture_img.size == heatmap_img.size:
        overlay = Image.blend(capture_img, heatmap_img, alpha=0.4)
        overlay_path = os.path.join(layer_dir, "overlay.jpg")
        overlay.save(overlay_path)
        logger.info("Saved overlay: %s", overlay_path)

    # Save render-over-capture overlay (alignment check)
    if capture_img.size == render_img.size:
        render_overlay = Image.blend(capture_img, render_img, alpha=0.5)
        render_overlay_path = os.path.join(layer_dir, "render_overlay.jpg")
        render_overlay.save(render_overlay_path)
        logger.info("Saved render overlay: %s", render_overlay_path)

    # Save 2x2 composite cropped to object bounding box:
    #   Capture       | Render
    #   Heatmap       | Render Overlay
    render_arr = np.array(render_img)
    nonblack = np.any(render_arr > 10, axis=2)
    if nonblack.any():
        ys, xs = np.where(nonblack)
        pad = 50  # pixels of context around the object
        x0 = max(int(xs.min()) - pad, 0)
        y0 = max(int(ys.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad, capture_img.width)
        y1 = min(int(ys.max()) + pad, capture_img.height)
        box = (x0, y0, x1, y1)

        cap_crop = capture_img.crop(box)
        rnd_crop = render_img.crop(box)
        hm_crop = Image.blend(capture_img, heatmap_img, alpha=0.4).crop(box) \
            if capture_img.size == heatmap_img.size else heatmap_img.crop(box)
        rnd_ov_crop = Image.blend(capture_img, render_img, alpha=0.5).crop(box) \
            if capture_img.size == render_img.size else rnd_crop

        cw, ch = cap_crop.size
        composite = Image.new("RGB", (cw * 2, ch * 2))
        composite.paste(cap_crop,    (0,  0))
        composite.paste(rnd_crop,    (cw, 0))
        composite.paste(hm_crop,     (0,  ch))
        composite.paste(rnd_ov_crop, (cw, ch))

        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw_comp = ImageDraw.Draw(composite)
        for text, lx, ly in [("Capture", 0, 0), ("Render", cw, 0),
                              ("Heatmap", 0, ch), ("Render Overlay", cw, ch)]:
            draw_comp.text((lx + 10, ly + 10), text,
                           fill=(255, 255, 255), font=font)

        comp_path = os.path.join(layer_dir, "composite.jpg")
        composite.save(comp_path)
        logger.info("Saved composite: %s (%dx%d, crop [%d:%d, %d:%d])",
                     comp_path, composite.width, composite.height, x0, x1, y0, y1)

    # Save individual patch crops (capture|render side-by-side per patch)
    patch_locations = metadata.get("patch_locations", [])
    scores_list = metadata.get("scores", [])
    if patch_locations:
        patches_dir = os.path.join(layer_dir, "patches")
        os.makedirs(patches_dir, exist_ok=True)
        for i, ((px, py, pw, ph), score) in enumerate(
                zip(patch_locations, scores_list)):
            cap_crop = capture_img.crop((px, py, px + pw, py + ph))
            rnd_crop = render_img.crop((px, py, px + pw, py + ph))
            pair = Image.new("RGB", (pw * 2, ph))
            pair.paste(cap_crop, (0, 0))
            pair.paste(rnd_crop, (pw, 0))
            score_val = float(score) if not isinstance(score, float) else score
            pair_path = os.path.join(
                patches_dir, f"patch_{i:03d}_s{score_val:.3f}.jpg")
            pair.save(pair_path)
        logger.info("Saved %d patch pairs to %s", len(patch_locations), patches_dir)

    # Save metadata JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata["timestamp"] = timestamp
    metadata["capture_path"] = capture_path
    metadata["render_path"] = render_path
    metadata["heatmap_path"] = heatmap_path
    json_path = os.path.join(layer_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata: %s", json_path)

    return save_dir
