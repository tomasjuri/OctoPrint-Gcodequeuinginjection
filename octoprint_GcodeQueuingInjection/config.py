
CAM_EXTRUDER_OFFSETS = {
    "x": -35,
    "y": 18,
    "z": 60,
}

RANDOM_OFFSET_RANGE = {
    "x": (-20,10),
    "y": (-20,10),
    "z": (-10,10),
}

RETRACTION_MM = 0.7
RETRACTION_SPEED = 1800
MOVE_FEEDRATE = 18000
WAIT_BEFORE_CAPTURE_MS = 2000

SNAPSHOT_URL = "http://127.0.0.1:8080/?action=snapshot"

CAPTURE_FOLDER = "~/data/layer_captures"

CAPTURE_EVERY_N_LAYERS = 5
CAPTURE_ALL_FIRST_N_LAYERS = 5

# --- Inference settings ---
# Fixed capture position: nozzle X/Y so camera ends up at bed center (125, 105)
# Given calibration offset (x:+31, y:-25): nozzle_x = 125-31 = 94, nozzle_y = 105+25 = 130
CAPTURE_NOZZLE_X = 94.0
CAPTURE_NOZZLE_Y = 130.0
# Nozzle Z offset above current layer height (center of CAM_EXTRUDER_OFFSETS.z + RANDOM_OFFSET_RANGE.z)
CAPTURE_Z_OFFSET = 60.0

# ONNX model path (FP32 by default for accuracy; FP16 variant also available)
ONNX_MODEL_PATH = "/home/tomasjurica/projects/PrusaSlicer_GcodeRenderer/render_matcher/resnet34_fixedVis_224-448_20260206_163236 copy/model_fp32.onnx"

# Calibration files
CALIBRATION_JSON_PATH = "/home/tomasjurica/projects/PrusaSlicer_GcodeRenderer/data/calibration.json"
CALIBRATION_NAME = "4.2.2026"
CAMERA_INTRINSIC_PATH = "/home/tomasjurica/projects/PrusaSlicer_GcodeRenderer/pygcode_viewer/camera_calib/camera_intrinsic.json"

# Patchwise inference
PATCH_SIZE = 448
PATCH_OVERLAP = 0.5
CNN_INPUT_SIZE = 224

# Render resolution: max dimension for the OpenGL render (lower = faster).
# The render is scaled up to match the capture before patch extraction.
RENDER_MAX_RESOLUTION = 2048

# Quick-check: number of center patches to test first.
# If all pass, skip the full patchwise check (big speedup for the common case).
QUICK_CHECK_PATCHES = 4

# Pass/fail thresholds
PASS_RATIO_THRESHOLD = 0.9   # 90% of patches must pass
PASS_SCORE_THRESHOLD = 0.5   # Individual patch score threshold

# Inference results save folder
INFERENCE_SAVE_FOLDER = "~/data/inference_results"

# Bed size (Prusa MK4)
BED_SIZE_X = 250.0
BED_SIZE_Y = 210.0
