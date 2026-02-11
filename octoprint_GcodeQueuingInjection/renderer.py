"""
GCode renderer wrapper for pygcode_viewer.

Renders what the camera should see at a given layer, using the same
camera pose as the physical setup (calibration offset + intrinsics).
"""

import json
import math
import logging
import os

logger = logging.getLogger(__name__)

# RPi 4 headless OpenGL setup (must be set before any EGL calls):
# - EGL_PLATFORM=drm: use DRM backend instead of X11 (no display server)
# - MESA_GL_VERSION_OVERRIDE=3.3: RPi V3D only supports GL 3.1 natively,
#   but Mesa can advertise 3.3 for compatibility (the missing features are
#   rarely used by libvgcode). MESA_GLSL_VERSION_OVERRIDE=330 matches.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("EGL_PLATFORM", "drm")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")


def load_calibration(calibration_json_path, calibration_name):
    """Load a named calibration entry from calibration.json."""
    with open(calibration_json_path) as f:
        calibrations = json.load(f)
    for calib in calibrations:
        if calib["name"] == calibration_name:
            return calib
    raise ValueError(f"Calibration '{calibration_name}' not found in {calibration_json_path}")


def _scaled_intrinsics(intrinsics_path, render_w, render_h):
    """Create a temporary intrinsics file scaled to the render resolution.

    Camera intrinsics (fx, fy, cx, cy) are in pixel coordinates for the
    original calibration resolution.  When rendering at a different size
    they must be scaled proportionally so the projection matches.

    Returns the path to use (original file if no scaling needed, or a
    temp file with adjusted values).
    """
    with open(intrinsics_path) as f:
        data = json.load(f)

    calib_w, calib_h = data["image_size"]
    if calib_w == render_w and calib_h == render_h:
        return intrinsics_path  # no scaling needed

    sx = render_w / calib_w
    sy = render_h / calib_h

    # Scale the 3x3 camera matrix
    mtx = data["camera_matrix"]
    mtx[0][0] *= sx   # fx
    mtx[0][2] *= sx   # cx
    mtx[1][1] *= sy   # fy
    mtx[1][2] *= sy   # cy
    data["camera_matrix"] = mtx
    data["image_size"] = [render_w, render_h]

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="intrinsics_", delete=False)
    json.dump(data, tmp, indent=2)
    tmp.close()

    logger.info("Scaled intrinsics %dx%d -> %dx%d (sx=%.4f, sy=%.4f): %s",
                calib_w, calib_h, render_w, render_h, sx, sy, tmp.name)
    return tmp.name


def render_layer(gcode_path, nozzle_pos, layer_n, layer_height,
                 calibration, intrinsics_path, image_size, output_path):
    """Render gcode up to layer_n from the camera pose.

    Args:
        gcode_path: Path to the .gcode file.
        nozzle_pos: dict with x, y, z of the nozzle at capture time.
        layer_n: Layer number to render up to.
        layer_height: Z height of the layer in mm.
        calibration: Calibration dict with 'position' and 'rotation' keys.
        intrinsics_path: Path to camera_intrinsic.json.
        image_size: Tuple of (width, height) for the output image.
        output_path: Where to save the rendered PNG.

    Returns:
        output_path on success.
    """
    import pygcode_viewer

    offset = calibration["position"]
    roll_deg = calibration["rotation"]["z"]

    # Camera position = nozzle + physical offset
    cam_x = nozzle_pos["x"] + offset["x"]
    cam_y = nozzle_pos["y"] + offset["y"]
    cam_z = nozzle_pos["z"] + offset["z"]

    # Up vector rotated by roll
    roll_rad = math.radians(roll_deg)
    up_x = -math.sin(roll_rad)
    up_y = math.cos(roll_rad)

    logger.info("Rendering layer %d (Z=%.2fmm) from camera (%.1f, %.1f, %.1f), roll=%.1f deg",
                layer_n, layer_height, cam_x, cam_y, cam_z, roll_deg)

    # Scale intrinsics to match the render resolution
    scaled_path = _scaled_intrinsics(intrinsics_path, image_size[0], image_size[1])

    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(str(gcode_path))
    viewer.set_intrinsics(str(scaled_path))

    viewer.set_camera(
        pos=(cam_x, cam_y, cam_z),
        target=(cam_x, cam_y + 0.001, 0.0),
        up=(up_x, up_y, 0.0),
    )
    viewer.set_near_far(1.0, 500.0)

    config = pygcode_viewer.ViewConfig()
    config.background_color = "#000000"
    config.width = image_size[0]
    config.height = image_size[1]
    viewer.set_config(config)
    viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#333333")
    viewer.set_layer_range(0, layer_n)

    viewer.render_to_file(str(output_path), require_intrinsics=True)

    # Clean up temp intrinsics file
    if scaled_path != intrinsics_path:
        try:
            os.unlink(scaled_path)
        except OSError:
            pass

    logger.info("Render saved to %s", output_path)
    return output_path
