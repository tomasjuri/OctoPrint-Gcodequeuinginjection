import octoprint.plugin
import logging
import random as rnd
from . import gcode_sequences as gcd
from . import calib_capture
import threading
import time
import numpy as np
import traceback

import re
from .config import (
    CAM_EXTRUDER_OFFSETS, 
    RANDOM_OFFSET_RANGE, 
    RETRACTION_MM, 
    RETRACTION_SPEED, 
    MOVE_FEEDRATE,
    WAIT_BEFORE_CAPTURE_MS,
    SNAPSHOT_URL,
    CAPTURE_FOLDER,
    CAPTURE_EVERY_N_LAYERS,
    CAPTURE_ALL_FIRST_N_LAYERS,
    CAPTURE_NOZZLE_X,
    CAPTURE_NOZZLE_Y,
    CAPTURE_Z_OFFSET,
    ONNX_MODEL_PATH,
    CALIBRATION_JSON_PATH,
    CALIBRATION_NAME,
    CAMERA_INTRINSIC_PATH,
    PATCH_SIZE,
    PATCH_OVERLAP,
    CNN_INPUT_SIZE,
    RENDER_MAX_RESOLUTION,
    QUICK_CHECK_PATCHES,
    PASS_RATIO_THRESHOLD,
    PASS_SCORE_THRESHOLD,
    INFERENCE_SAVE_FOLDER,
    BED_SIZE_X,
    BED_SIZE_Y,
)
from .camera import Camera
from datetime import datetime

import json
import os

class GcodequeuinginjectionPlugin(
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.SimpleApiPlugin,
    octoprint.plugin.EventHandlerPlugin
):
    def __init__(self):
        self.print_gcode = False
        self.state = None

        self._logger = logging.getLogger(__name__)
        
        # Position tracking
        self._position_signal = threading.Event()
        self._position_signal.set()
        self._position_request_sent = False
        self._position_payload = None
        self._position_timeout = 30.0
        
        # Capture completion tracking
        self._capture_signal = threading.Event()
        self._capture_signal.set()
        self._capture_request_sent = False
        self._capture_timeout = 30.0
        
        self.camera = Camera()
        self.print_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Inference (lazy-loaded on first use)
        self._inference_session = None
        self._calibration = None


    def get_settings_defaults(self):
        """Return the default settings for the plugin using values from config.py"""
        return {
            "cam_extruder_offsets": CAM_EXTRUDER_OFFSETS,
            "random_offset_range": {
                "x": list(RANDOM_OFFSET_RANGE["x"]),
                "y": list(RANDOM_OFFSET_RANGE["y"]), 
                "z": list(RANDOM_OFFSET_RANGE["z"]),
            },
            "retraction_mm": RETRACTION_MM,
            "retraction_speed": RETRACTION_SPEED,
            "move_feedrate": MOVE_FEEDRATE,
            "wait_before_capture_ms": WAIT_BEFORE_CAPTURE_MS,
            "snapshot_url": SNAPSHOT_URL,
            "capture_folder": CAPTURE_FOLDER,
            "capture_every_n_layers": CAPTURE_EVERY_N_LAYERS,
            "capture_all_first_n_layers": CAPTURE_ALL_FIRST_N_LAYERS,
            # Inference settings
            "capture_nozzle_x": CAPTURE_NOZZLE_X,
            "capture_nozzle_y": CAPTURE_NOZZLE_Y,
            "capture_z_offset": CAPTURE_Z_OFFSET,
            "onnx_model_path": ONNX_MODEL_PATH,
            "calibration_json_path": CALIBRATION_JSON_PATH,
            "calibration_name": CALIBRATION_NAME,
            "camera_intrinsic_path": CAMERA_INTRINSIC_PATH,
            "patch_size": PATCH_SIZE,
            "patch_overlap": PATCH_OVERLAP,
            "cnn_input_size": CNN_INPUT_SIZE,
            "render_max_resolution": RENDER_MAX_RESOLUTION,
            "quick_check_patches": QUICK_CHECK_PATCHES,
            "pass_ratio_threshold": PASS_RATIO_THRESHOLD,
            "pass_score_threshold": PASS_SCORE_THRESHOLD,
            "inference_save_folder": INFERENCE_SAVE_FOLDER,
            "bed_size_x": BED_SIZE_X,
            "bed_size_y": BED_SIZE_Y,
        }

    def on_settings_save(self, data):
        """Handle settings save from UI"""
        octoprint.plugin.SettingsPlugin.on_settings_save(self, data)
        self._logger.info("Settings updated via UI")

    def _get_save_path(self):
        """Get the configured save path"""
        save_path = os.path.expanduser(self._settings.get(["capture_folder"]))
        save_path = os.path.join(save_path, self.print_timestamp)
        if not os.path.exists(save_path):
            self._logger.debug("Creating capture folder: %s", save_path)
            os.makedirs(save_path)
        return save_path

    def _get_save_path_chessboard(self):
        """Get the configured save path for the chessboard calibration"""
        save_path = os.path.expanduser(self._settings.get(["capture_folder"]))
        save_path = os.path.join(save_path, "chessboard", self.print_timestamp)
        if not os.path.exists(save_path):
            self._logger.debug("Creating chessboard capture folder: %s", save_path)
            os.makedirs(save_path)
        return save_path
    
    def _get_save_path_aruco(self):
        """Get the configured save path for the aruco calibration"""
        save_path = os.path.expanduser(self._settings.get(["capture_folder"]))
        save_path = os.path.join(save_path, "aruco", self.print_timestamp)
        if not os.path.exists(save_path):
            self._logger.debug("Creating aruco capture folder: %s", save_path)
            os.makedirs(save_path)
        return save_path

    def gen_capture_pos(self, cmd, current_position):
        """Generate fixed capture position for inference.

        Nozzle moves to configured X/Y (so camera ends up at bed center)
        and Z = current layer Z + configured offset.

        Args:
            cmd: M240 command string (e.g. "M240 Z0.4 ZN1 S0").
            current_position: Current nozzle position dict with x, y, z.

        Returns:
            Tuple of (capture_position dict, layer_n, layer_height).
        """
        # Parse layer info from M240 Z<height> ZN<layer_num> S<state>
        parts = cmd.split(" ")
        layer_height = float(parts[1][1:])  # "Z0.4" -> 0.4
        layer_n = int(parts[2][2:])          # "ZN1" -> 1

        nozzle_x = self._safe_float(self._settings.get(["capture_nozzle_x"]), CAPTURE_NOZZLE_X)
        nozzle_y = self._safe_float(self._settings.get(["capture_nozzle_y"]), CAPTURE_NOZZLE_Y)
        z_offset = self._safe_float(self._settings.get(["capture_z_offset"]), CAPTURE_Z_OFFSET)

        capture_position = {
            "x": nozzle_x,
            "y": nozzle_y,
            "z": current_position["z"] + z_offset,
        }
        self._logger.info("Capture position: nozzle X=%.1f Y=%.1f Z=%.1f (layer %d, Z_layer=%.2f + offset=%.1f)",
                          nozzle_x, nozzle_y, capture_position["z"], layer_n, current_position["z"], z_offset)
        return capture_position, layer_n, layer_height

    def parse_position_line(self, line):
        """Parse M114 response"""
        regex_float_pattern = r"[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?"
        regex_position = re.compile(
            r"X:\s*(?P<x>{float})\s*Y:\s*(?P<y>{float})\s*Z:\s*(?P<z>{float})\s*(?:E:\s*(?P<e>{float}))?"
            .format(float=regex_float_pattern))
        
        match = regex_position.search(line)
        if match is not None:
            e_value = match.group("e")
            result = {
                'x': float(match.group("x")),
                'y': float(match.group("y")),
                'z': float(match.group("z")),
                'e': float(e_value) if e_value is not None else 0.0
            }
            return result
        return None

    def get_position_async(self, timeout=None):
        """Request position"""
        if timeout is None:
            timeout = self._position_timeout
            
        self._logger.debug("Requesting position asynchronously")
        # Warning: we can only request one position at a time!
        if not self._position_signal.is_set():
            self._logger.warning("Position request already in progress")
            return None
            
        # Clear previous state
        self._position_payload = None
        self._position_request_sent = True
        self._position_signal.clear()
        
        # Send standard position request
        tags = {'plugin:GcodeQueuingInjection', 'position-request'}
        self._printer.commands(["M400", "M114"], tags=tags)
        
        # Wait for response with timeout
        event_is_set = self._position_signal.wait(timeout)
        if not event_is_set:
            self._logger.warning("Timeout occurred while requesting position")
            return None
            
        return self._position_payload

    def wait_for_capture_completion(self, timeout=None):
        """Wait for capture completion signal"""
        if timeout is None:
            timeout = self._capture_timeout
            
        self._logger.debug("Waiting for capture completion")
        
        # Check if we can wait (should be clear when capture starts)
        if self._capture_signal.is_set():
            self._logger.warning("Capture signal is already set, this might indicate a timing issue")
            
        # Wait for capture completion signal with timeout
        event_is_set = self._capture_signal.wait(timeout)
        if not event_is_set:
            self._logger.warning("Timeout occurred while waiting for capture completion")
            return False
            
        self._logger.debug("Capture completion confirmed")
        return True

    def on_position_received(self, payload):
        """Handle position response"""
        if self._position_request_sent:
            self._position_request_sent = False
            self._logger.debug("Position received: %s", payload)
            self._position_payload = payload
            self._position_signal.set()
        else:
            self._logger.debug("Position response received but not requested by us, ignoring")
    
    def get_gcode_path(self):
        # Try to resolve current printed gcode path from OctoPrint's current job info
        gcode_path = "NoGcodeName"
        try:
            job = self._printer.get_current_job()
            if isinstance(job, dict):
                file_info = job.get("file") or {}
                # Common fields: path (within uploads), name, display, origin
                gcode_path = file_info.get("path") or file_info.get("name") or "NoGcodeName"
        except Exception as e:
            self._logger.warning("Failed to fetch current job info for metadata: %s", e)
        return gcode_path

    def capture_img(self, capture_position, layer_n, layer_height, save_path=None, filename=None):
        """Capture image at specified position and save with metadata."""
        self._logger.debug("Capturing image. Expected to be at position: %s", capture_position)
        
        img = self.camera.capture_image(self._settings.get(["snapshot_url"]))
        self._logger.debug("Image captured")
        self._capture_signal.set()

        x = int(capture_position['x'])
        y = int(capture_position['y'])
        z = int(capture_position['z'])
        layer_n = 0 if layer_n is None else int(layer_n)
        layer_height = 0 if layer_height is None else np.round(float(layer_height), 2)
        
        gcode_path = self.get_gcode_path()

        img_filepath = None
        json_filepath = None
        if filename is None:
            img_filepath = f"img_ZN{layer_n}_Z{layer_height}_X{x}_Y{y}_Z{z}.jpg"
            json_filepath = f"img_ZN{layer_n}_Z{layer_height}_X{x}_Y{y}_Z{z}.json"
        else:
            img_filepath = filename
            json_filepath = filename + ".json"
        
        self._logger.debug(f"Generating image name: {img_filepath}")
        
        if save_path is None:
            save_path = self._get_save_path()
        path, folder = os.path.split(save_path)
        save_path = os.path.join(path, f"{gcode_path}_{folder}")
        os.makedirs(save_path, exist_ok=True)
        im_path = os.path.join(save_path, img_filepath)
        self._logger.debug("Saving image to %s", im_path)
        img.save(im_path)
        self._logger.debug("Image saved to %s", im_path)
        
        data = {
            "img_filepath": img_filepath,
            "json_filepath": json_filepath,
            "capture_position": capture_position,
            "layer_n": layer_n,
            "layer_height": layer_height,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "gcode_path": gcode_path,
            # Calibration placeholders (to be populated by future calibration feature)
            "calibration": {
                "data": None,
                "timestamp": None,
            },
        }
        json_path = os.path.join(save_path, json_filepath)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        self._logger.debug("Capture complete, saved to %s and %s", im_path, json_path)
        

    def _safe_float(self, value, default=0.0):
        """Safely convert a value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value, default=0):
        """Safely convert a value to int."""
        try:
            return int(float(value))  # Handle string floats like "500.0"
        except (TypeError, ValueError):
            return default

    def _get_validated_settings(self):
        """Get and validate settings, merging with defaults for missing values."""
        # Get cam_extruder_offsets - merge with defaults for any missing keys
        offsets = self._settings.get(["cam_extruder_offsets"])
        if not isinstance(offsets, dict):
            offsets = {}
        
        # Merge user values with defaults, converting strings to floats
        offsets = {
            'x': self._safe_float(offsets.get('x'), CAM_EXTRUDER_OFFSETS['x']),
            'y': self._safe_float(offsets.get('y'), CAM_EXTRUDER_OFFSETS['y']),
            'z': self._safe_float(offsets.get('z'), CAM_EXTRUDER_OFFSETS['z']),
        }
        
        # Get random_offset_range - merge with defaults for any missing keys
        rnd_range = self._settings.get(["random_offset_range"])
        if not isinstance(rnd_range, dict):
            rnd_range = {}
        
        # Convert string values to floats in the range lists
        def convert_range(val, default):
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return [self._safe_float(val[0], default[0]), self._safe_float(val[1], default[1])]
            return list(default)
        
        # Merge user values with defaults
        rnd_range = {
            'x': convert_range(rnd_range.get('x'), RANDOM_OFFSET_RANGE['x']),
            'y': convert_range(rnd_range.get('y'), RANDOM_OFFSET_RANGE['y']),
            'z': convert_range(rnd_range.get('z'), RANDOM_OFFSET_RANGE['z']),
        }
        
        return offsets, rnd_range

    def _get_wait_before_capture_ms(self):
        """Get wait_before_capture_ms with type conversion."""
        return self._safe_int(self._settings.get(["wait_before_capture_ms"]), WAIT_BEFORE_CAPTURE_MS)

    def _get_retraction_mm(self):
        """Get retraction_mm with type conversion."""
        return self._safe_float(self._settings.get(["retraction_mm"]), RETRACTION_MM)

    def _get_retraction_speed(self):
        """Get retraction_speed with type conversion."""
        return self._safe_float(self._settings.get(["retraction_speed"]), RETRACTION_SPEED)

    def _get_move_feedrate(self):
        """Get move_feedrate with type conversion."""
        return self._safe_float(self._settings.get(["move_feedrate"]), MOVE_FEEDRATE)

    def _get_capture_every_n_layers(self):
        """Get capture_every_n_layers with type conversion. Minimum 1."""
        val = self._safe_int(self._settings.get(["capture_every_n_layers"]), CAPTURE_EVERY_N_LAYERS)
        return max(1, val)

    def _get_capture_all_first_n_layers(self):
        """Get capture_all_first_n_layers with type conversion. Minimum 0."""
        val = self._safe_int(self._settings.get(["capture_all_first_n_layers"]), CAPTURE_ALL_FIRST_N_LAYERS)
        return max(0, val)

    def capture_sequence_async(self, original_cmd):
        """Handle the complete capture sequence asynchronously - simplified using position sync"""
        def capture_worker():
            capture_thread = None
            inference_capture = None
            try:
                self._logger.debug("Started async capture worker")
                
                # Get initial position - this also serves as sync point
                start_position = self.get_position_async()
                if start_position is None:
                    self._logger.error("Failed to get position, aborting capture")
                    return
                
                # Validate position has required keys
                if not isinstance(start_position, dict) or not all(k in start_position for k in ('x', 'y', 'z')):
                    self._logger.error("Invalid position data: %s, aborting capture", start_position)
                    return
                
                # Generate fixed capture position for inference
                capture_pos, layer_n, layer_height = self.gen_capture_pos(
                    original_cmd, start_position)
                
                # 1. Send movement commands  
                move_commands = gcd.gen_move_to_capture_gcode(
                    capture_position=capture_pos,
                    retraction_mm=self._get_retraction_mm(),
                    retraction_speed=self._get_retraction_speed(),
                )
                self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                
                # 2. Wait for moves to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to reach capture position, aborting")
                    return
                
                # 3. Prepare for capture completion signaling
                self._capture_signal.clear()  # Clear signal before starting capture
                
                # 4. Wait for wait_before_capture_ms
                wait_ms = self._get_wait_before_capture_ms()
                self._logger.debug(f"Waiting for {wait_ms} ms before capture")
                time.sleep(wait_ms / 1000)
                
                # 5. Capture image - printer is now guaranteed to be in position
                capture_thread = threading.Thread(
                    target=self.capture_img, args=(capture_pos, layer_n, layer_height))
                capture_thread.start()
                
                # 6. Wait for capture completion signal instead of fixed delay
                if not self.wait_for_capture_completion():
                    self._logger.error("Capture completion timeout, continuing anyway")

                # 6b. Take the inference snapshot NOW while nozzle is still stationary
                self._logger.info("Capturing inference image (nozzle stationary)...")
                inference_capture = self.camera.capture_image(
                    self._settings.get(["snapshot_url"]))
                self._logger.info("Inference capture: %dx%d", *inference_capture.size)

                # 7. Send return commands after capture is confirmed complete
                return_commands = gcd.gen_capture_and_return_gcode(
                    return_position=start_position,
                    retraction_mm=self._get_retraction_mm(),
                    retraction_speed=self._get_retraction_speed(),
                )
                self._printer.commands(return_commands, tags={'plugin:GcodeQueuingInjection', 'capture-return'})
                
                # 8. Wait for return sequence to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to complete return sequence")
                
                self._logger.debug("Capture sequence completed successfully")
                    
            except Exception as e:
                self._logger.error("Error in capture sequence: %s\n%s", e, traceback.format_exc())
            finally:
                # Wait for image capture to complete BEFORE releasing job hold
                if capture_thread:
                    capture_thread.join(timeout=10.0)
                    self._logger.debug("Image capture thread completed")
                
                # Always release the job hold LAST
                self._printer.set_job_on_hold(False)
                self._logger.debug("Released job hold lock")

                # Run inference in background so it doesn't block printing.
                # inference_capture was taken while the nozzle was stationary.
                if inference_capture is not None:
                    def _inference_bg(img=inference_capture):
                        try:
                            passed = self._run_inference(
                                capture_pos, layer_n, layer_height, capture_img=img)
                        except Exception as exc:
                            self._logger.error("Background inference error: %s\n%s",
                                               exc, traceback.format_exc())
                            return

                        if not passed:
                            try:
                                self._logger.info("INFERENCE FAILED - pausing print at layer %d", layer_n)
                                self._printer.pause_print()
                                # Wait for pause to take effect before parking
                                self._logger.info("Waiting for pause to take effect...")
                                for i in range(30):  # up to 15 seconds
                                    time.sleep(0.5)
                                    if self._printer.is_paused():
                                        self._logger.info("Printer paused after %.1fs", (i + 1) * 0.5)
                                        break
                                self._park_nozzle(layer_height)
                            except Exception as exc:
                                self._logger.error("Error during pause/park: %s\n%s",
                                                   exc, traceback.format_exc())

                    inference_thread = threading.Thread(target=_inference_bg, daemon=True)
                    inference_thread.start()

        
        # Start the worker thread
        worker_thread = threading.Thread(target=capture_worker)
        worker_thread.daemon = True
        worker_thread.start()

    # --- Inference pipeline ---

    def _get_inference_session(self):
        """Lazy-load the ONNX inference session."""
        if self._inference_session is None:
            from .inference import InferenceSession
            model_path = os.path.expanduser(self._settings.get(["onnx_model_path"]))
            self._inference_session = InferenceSession(model_path)
        return self._inference_session

    def _get_calibration(self):
        """Lazy-load calibration data."""
        if self._calibration is None:
            from .renderer import load_calibration
            calib_path = self._settings.get(["calibration_json_path"])
            calib_name = self._settings.get(["calibration_name"])
            self._calibration = load_calibration(calib_path, calib_name)
            self._logger.info("Loaded calibration '%s' from %s", calib_name, calib_path)
        return self._calibration

    def _get_gcode_full_path(self):
        """Resolve absolute filesystem path of the currently printing gcode."""
        try:
            job = self._printer.get_current_job()
            if isinstance(job, dict):
                file_info = job.get("file") or {}
                rel_path = file_info.get("path") or file_info.get("name")
                if rel_path:
                    full_path = self._file_manager.path_on_disk("local", rel_path)
                    return full_path
        except Exception as e:
            self._logger.warning("Could not resolve gcode path: %s", e)
        return None

    def _run_inference(self, capture_pos, layer_n, layer_height, capture_img=None):
        """Run the full inference pipeline: render + CNN + save.

        Two-phase strategy for speed:
          1. Quick check: a few patches from the object center.
             If all pass -> PASS immediately (fast path, ~2-3s).
          2. Full check: all overlapping patches (only when quick check fails).

        Args:
            capture_img: PIL Image taken while nozzle was stationary at capture
                         position.  If None a new snapshot is taken (fallback).

        Returns True if the print should continue (pass), False to pause (fail).
        """
        try:
            from .renderer import render_layer
            from .inference import extract_patches, extract_center_patches
            from .visualizations import create_heatmap
            from PIL import Image
            import tempfile

            self._logger.info("=== Starting inference for layer %d ===", layer_n)
            t_start = time.time()

            # 1. Use the pre-captured image (taken while stationary)
            if capture_img is None:
                self._logger.warning("No pre-captured image, taking snapshot now (may be blurred)")
                capture_img = self.camera.capture_image(self._settings.get(["snapshot_url"]))
            capture_w, capture_h = capture_img.size
            self._logger.info("Capture image: %dx%d", capture_w, capture_h)

            # 2. Render at lower resolution for speed, then scale up
            gcode_path = self._get_gcode_full_path()
            if gcode_path is None:
                self._logger.error("Cannot resolve gcode path, skipping inference")
                return True

            calibration = self._get_calibration()
            intrinsics_path = self._settings.get(["camera_intrinsic_path"])

            max_res = self._safe_int(
                self._settings.get(["render_max_resolution"]), RENDER_MAX_RESOLUTION)
            MAX_TEX = 4096  # GPU hard limit
            max_dim = min(max_res, MAX_TEX)

            render_w, render_h = capture_w, capture_h
            if render_w > max_dim or render_h > max_dim:
                scale = max_dim / max(render_w, render_h)
                render_w = int(render_w * scale)
                render_h = int(render_h * scale)
            self._logger.info("Render resolution: %dx%d (max %d)", render_w, render_h, max_dim)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                render_path = tmp.name

            render_layer(
                gcode_path=gcode_path,
                nozzle_pos=capture_pos,
                layer_n=layer_n,
                layer_height=layer_height,
                calibration=calibration,
                intrinsics_path=intrinsics_path,
                image_size=(render_w, render_h),
                output_path=render_path,
            )
            render_img = Image.open(render_path).convert("RGB")
            if render_img.size != (capture_w, capture_h):
                render_img = render_img.resize((capture_w, capture_h), Image.LANCZOS)
                self._logger.info("Scaled render %dx%d -> %dx%d",
                                  render_w, render_h, capture_w, capture_h)
            t_render = time.time() - t_start
            self._logger.info("Render done in %.1fs", t_render)

            # Read settings
            patch_size = self._safe_int(self._settings.get(["patch_size"]), PATCH_SIZE)
            patch_overlap = self._safe_float(self._settings.get(["patch_overlap"]), PATCH_OVERLAP)
            cnn_input_size = self._safe_int(self._settings.get(["cnn_input_size"]), CNN_INPUT_SIZE)
            ratio_threshold = self._safe_float(self._settings.get(["pass_ratio_threshold"]), PASS_RATIO_THRESHOLD)
            score_threshold = self._safe_float(self._settings.get(["pass_score_threshold"]), PASS_SCORE_THRESHOLD)
            n_quick = self._safe_int(self._settings.get(["quick_check_patches"]), QUICK_CHECK_PATCHES)

            session = self._get_inference_session()

            # --- Phase 1: Quick center check ---
            src_q, rnd_q, locs_q = extract_center_patches(
                capture_img, render_img, patch_size, cnn_input_size, n_patches=n_quick)

            quick_pass = False
            quick_all_fail = False
            if len(locs_q) > 0:
                scores_q = session.run(src_q, rnd_q)
                quick_pass = bool(np.all(scores_q >= score_threshold))
                quick_all_fail = bool(np.all(scores_q < score_threshold))
                self._logger.info("Quick check: %d center patches, all pass=%s (scores: %s)",
                                  len(locs_q), quick_pass,
                                  ", ".join(f"{s:.3f}" for s in scores_q))
            else:
                self._logger.warning("No center patches found, proceeding to full check")

            if quick_pass:
                # Fast path: all center patches pass -> PASS immediately
                elapsed = time.time() - t_start
                self._logger.info("=== Inference complete: PASS (quick, %.1fs) -- layer %d ===",
                                  elapsed, layer_n)
                heatmap_img = create_heatmap((capture_w, capture_h), locs_q, scores_q)
                self._save_inference_results(
                    capture_img, render_img, heatmap_img, capture_pos, layer_n,
                    layer_height, True, scores_q, locs_q,
                    {"n_patches": len(locs_q), "n_passing": len(locs_q),
                     "pass_ratio": 1.0, "mean_score": float(scores_q.mean()),
                     "min_score": float(scores_q.min()),
                     "max_score": float(scores_q.max()), "quick_pass": True})
                self._cleanup_temp(render_path)
                return True

            if quick_all_fail:
                # Fast fail: all center patches bad -> FAIL immediately
                n_passing = int((scores_q >= score_threshold).sum())
                ratio = n_passing / len(scores_q)
                elapsed = time.time() - t_start
                self._logger.info("=== Inference complete: FAIL (quick, %.1fs) -- layer %d, "
                                  "all %d center patches failed ===", elapsed, layer_n, len(locs_q))
                heatmap_img = create_heatmap((capture_w, capture_h), locs_q, scores_q)
                self._save_inference_results(
                    capture_img, render_img, heatmap_img, capture_pos, layer_n,
                    layer_height, False, scores_q, locs_q,
                    {"n_patches": len(locs_q), "n_passing": n_passing,
                     "pass_ratio": float(ratio),
                     "mean_score": float(scores_q.mean()),
                     "min_score": float(scores_q.min()),
                     "max_score": float(scores_q.max()), "quick_fail": True})
                self._cleanup_temp(render_path)
                return False

            # --- Phase 2: Full patchwise check ---
            self._logger.info("Quick check mixed results, running full patchwise inference...")
            source_batch, render_batch, patch_locations = extract_patches(
                capture_img, render_img, patch_size, patch_overlap, cnn_input_size)

            if len(patch_locations) == 0:
                self._logger.warning("No valid patches, skipping inference (render may be empty)")
                self._cleanup_temp(render_path)
                return True

            scores = session.run(source_batch, render_batch)
            passed, ratio, stats = session.decide(scores, ratio_threshold, score_threshold)

            # Save full results
            heatmap_img = create_heatmap((capture_w, capture_h), patch_locations, scores)
            self._save_inference_results(
                capture_img, render_img, heatmap_img, capture_pos, layer_n,
                layer_height, passed, scores, patch_locations, stats)

            elapsed = time.time() - t_start
            verdict = "PASS" if passed else "FAIL"
            self._logger.info("=== Inference complete: %s (full, %.1fs) -- layer %d, %.1f%% patches passed ===",
                              verdict, elapsed, layer_n, ratio * 100)

            self._cleanup_temp(render_path)
            return passed

        except Exception as e:
            self._logger.error("Inference error (continuing print): %s\n%s", e, traceback.format_exc())
            return True  # Don't pause on inference errors

    def _get_inference_run_dir(self):
        """Get the per-print inference results directory.

        Uses print_timestamp + gcode name so every layer of the same print
        lands in the same folder.
        """
        save_folder = os.path.expanduser(
            self._settings.get(["inference_save_folder"]))
        gcode_name = self.get_gcode_path()
        run_dir = os.path.join(
            save_folder, f"{self.print_timestamp}_{gcode_name}")
        return run_dir

    def _save_inference_results(self, capture_img, render_img, heatmap_img,
                                capture_pos, layer_n, layer_height,
                                passed, scores, patch_locations, stats):
        """Save inference outputs (images + metadata JSON)."""
        from .visualizations import save_results

        gcode_name = self.get_gcode_path()
        metadata = {
            "layer_n": layer_n,
            "layer_height": float(layer_height),
            "capture_position": capture_pos,
            "gcode_name": gcode_name,
            "passed": passed,
            "stats": stats,
            "scores": scores.tolist() if hasattr(scores, 'tolist') else list(scores),
            "patch_locations": patch_locations,
        }
        save_folder = self._settings.get(["inference_save_folder"])
        run_dir = self._get_inference_run_dir()
        save_results(capture_img, render_img, heatmap_img, metadata,
                     save_folder, run_dir=run_dir)

        # Overwrite latest/ folder inside the run directory
        self._update_latest_images(run_dir, capture_img, render_img, heatmap_img)

    def _update_latest_images(self, run_dir, capture_img, render_img, heatmap_img):
        """Overwrite run_dir/latest/ with the current overlay and a 2x2 composite.

        The composite is cropped to the object bounding box and arranged as:
            Capture       | Render
            Heatmap       | Render Overlay
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            latest_dir = os.path.join(run_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            # --- plain overlay (heatmap over capture) ---
            if capture_img.size == heatmap_img.size:
                overlay = Image.blend(capture_img, heatmap_img, alpha=0.4)
            else:
                overlay = heatmap_img
            overlay.save(os.path.join(latest_dir, "latest.jpg"), quality=85)

            # --- 2x2 composite cropped to object bbox ---
            render_arr = np.array(render_img)
            nonblack = np.any(render_arr > 10, axis=2)
            if not nonblack.any():
                self._logger.info("Saved latest overlay (no object pixels for composite)")
                return

            ys, xs = np.where(nonblack)
            pad = 50
            x0 = max(int(xs.min()) - pad, 0)
            y0 = max(int(ys.min()) - pad, 0)
            x1 = min(int(xs.max()) + pad, capture_img.width)
            y1 = min(int(ys.max()) + pad, capture_img.height)
            box = (x0, y0, x1, y1)

            cap_crop = capture_img.crop(box)
            rnd_crop = render_img.crop(box)
            hm_crop  = overlay.crop(box)  # heatmap-over-capture crop
            if capture_img.size == render_img.size:
                rnd_ov_crop = Image.blend(capture_img, render_img, alpha=0.5).crop(box)
            else:
                rnd_ov_crop = rnd_crop

            cw, ch = cap_crop.size
            matrix = Image.new("RGB", (cw * 2, ch * 2))
            matrix.paste(cap_crop,    (0,  0))
            matrix.paste(rnd_crop,    (cw, 0))
            matrix.paste(hm_crop,     (0,  ch))
            matrix.paste(rnd_ov_crop, (cw, ch))

            # Labels
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw = ImageDraw.Draw(matrix)
            labels = [("Capture", 0, 0), ("Render", cw, 0),
                      ("Heatmap", 0, ch), ("Render Overlay", cw, ch)]
            for text, lx, ly in labels:
                draw.text((lx + 10, ly + 10), text,
                          fill=(255, 255, 255), font=font)

            matrix.save(os.path.join(latest_dir, "composite.jpg"), quality=85)
            self._logger.info("Saved latest overlay + composite to %s", latest_dir)
        except Exception as exc:
            self._logger.error("Failed to save latest images: %s", exc)

    def _park_nozzle(self, layer_height):
        """Park the nozzle at a safe corner position after pausing.

        Retracts filament, lifts Z 60mm above the last layer, and moves
        to the front-left corner of the bed.
        """
        park_z = float(layer_height) + 60.0
        park_x = 0.0
        park_y = self._safe_float(self._settings.get(["bed_size_y"]), BED_SIZE_Y)
        retraction_mm = self._get_retraction_mm()
        retraction_speed = self._get_retraction_speed()

        park_commands = [
            "M83",                                         # relative extrusion
            f"G1 E-{retraction_mm} F{retraction_speed}",  # retract
            "G90",                                         # absolute positioning
            f"G0 Z{park_z:.1f} F600",                     # lift Z first
            f"G0 X{park_x:.1f} Y{park_y:.1f} F{MOVE_FEEDRATE}",  # move to corner
            "M400",                                        # wait for moves
        ]
        self._logger.info("Sending park commands (force=True): Z=%.1f X=%.0f Y=%.0f",
                          park_z, park_x, park_y)
        self._printer.commands(park_commands,
                               tags={'plugin:GcodeQueuingInjection', 'park-nozzle'},
                               force=True)
        self._logger.info("Parked nozzle at X=%.0f Y=%.0f Z=%.1f after inference failure",
                          park_x, park_y, park_z)

    @staticmethod
    def _cleanup_temp(path):
        """Remove a temporary file, ignoring errors."""
        try:
            os.unlink(path)
        except OSError:
            pass

    def _should_capture_layer(self, layer_n):
        """Determine whether to capture on this layer based on frequency settings.
        
        Captures every layer for the first N layers, then only every Mth layer.
        """
        first_n = self._get_capture_all_first_n_layers()
        every_n = self._get_capture_every_n_layers()
        
        if layer_n <= first_n:
            return True
        return layer_n % every_n == 0

    def gcode_queuing(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        """Handle gcode queuing."""
        if gcode and gcode == gcd.CAPTURE_GCODE[0]:
            self.state = [c for c in cmd.split(" ") if c.startswith("S")][0]
            
            # S0: Start capture sequence using Octolapse pattern
            if self.state == "S0": 
                # Parse layer number from command (format: M240 Z<height> ZN<layer_num> S<state>)
                parts = cmd.split(" ")
                layer_n = int(parts[2][2:])  # Extract from "ZN<num>"
                
                # Check if we should capture this layer
                if not self._should_capture_layer(layer_n):
                    self._logger.debug("Skipping capture for layer %d (capture_every_n=%d, first_n=%d)",
                                       layer_n, self._get_capture_every_n_layers(), self._get_capture_all_first_n_layers())
                    return None,
                
                self._logger.debug("S0 STATE: Starting capture sequence with command: %s", cmd)
                
                # Use job_on_hold to pause queue processing
                if self._printer.set_job_on_hold(True):
                    self._logger.debug("Job on hold acquired, starting the sequence")
                    self.capture_sequence_async(cmd)
                    return None,
                else:
                    self._logger.warning("Could not set job on hold, falling back")
                    return None,
            
            return cmd

    
    def on_event(self, event, payload):
        """Reset print timestamp on every print start so restarts get a new folder."""
        if event == "PrintStarted":
            self.print_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._logger.info("Print started — new timestamp: %s", self.print_timestamp)

    def gcode_sent(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        """Handle gcode sent."""
        if cmd and cmd == gcd.START_PRINT_GCODE[0]:
            self.print_gcode = True
            self.print_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cmd and cmd == gcd.STOP_PRINT_GCODE[0]:
            self.print_gcode = False
        
        if self.print_gcode:
            self._logger.debug("Gcode sent: %s", cmd)
            


    def gcode_received(self, comm_instance, line, *args, **kwargs):
        """Handle M114 position responses using Octolapse pattern"""
        
        # Only process if we requested the position (Octolapse pattern)
        if self._position_request_sent and 'X:' in line and 'Y:' in line and 'Z:' in line:
            self._logger.debug("Processing position response: %s", line)
            
            position = self.parse_position_line(line)
            if position:
                self.on_position_received(position)
            else:
                self._logger.warning("Failed to parse position from line: %s", line)
        
        return line


    # http://localhost:8060/api/plugin/GcodeQueuingInjection?command=calib_capture_chessboard
    # http://localhost:8060/api/plugin/GcodeQueuingInjection?command=calib_capture_aruco
    def on_api_get(self, request):
        """Handle GET requests for direct URL access"""
        command = request.args.get("command")
        self._logger.info(f"API GET request received for command: {command}")
        self._logger.info("🎯 CALIBRATION CAPTURE TRIGGERED via GET! Starting calibration sequence...")
        
        if command == "calib_capture_chessboard":
            self._logger.info("Capturing chessboard calibration images...")
            self.calib_capture_sequence_async(save_path=self._get_save_path_chessboard())
            return dict(success=True, message="Calibration capture executed via GET", result=True)
        
        elif command == "calib_capture_aruco":
            self._logger.info("Capturing aruco calibration images...")
            self.calib_capture_sequence_async(save_path=self._get_save_path_aruco())
            return dict(success=True, message="Calibration capture executed via GET", result=True)
        
        elif command == "calib_position":
            self._logger.info("Getting calibration position...")
            self.move_to_calib_position()
            return dict(success=True, message="Calibration position retrieved via GET", result=True)

        elif command == "capture":
            self._logger.info("Capturing image...")
            filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
            self.api_run_capture(save_path=self._get_save_path_chessboard(), filename=filename)
            return dict(success=True, message="Capture executed via GET", result=True)
        
        else:
            return dict(success=False, error=f"Unknown GET command: {command}")


    def is_api_protected(self):
        """Allow unauthenticated access to API for direct URL access"""
        return False

    def get_template_configs(self):
        """Define which templates the plugin provides."""
        return [
            dict(type="settings", custom_bindings=False)
        ]

    def get_assets(self):
        """Define plugin's asset files to automatically include in the core UI."""
        return {
            "js": ["js/GcodeQueuingInjection.js"],
            "css": ["css/GcodeQueuingInjection.css"],
            "less": ["less/GcodeQueuingInjection.less"]
        }

    ##~~ Softwareupdate hook

    def get_update_information(self):
        """Define the configuration for your plugin to use with the Software Update Plugin."""
        return {
            "GcodeQueuingInjection": {
                "displayName": "Gcodequeuinginjection Plugin",
                "displayVersion": self._plugin_version,

                # version check: github repository
                "type": "github_release",
                "user": "tomasjuri",
                "repo": "OctoPrint-Gcodequeuinginjection",
                "current": self._plugin_version,

                # update method: pip
                "pip": "https://github.com/tomasjuri/OctoPrint-Gcodequeuinginjection/archive/{target_version}.zip",
            }
        }


    def calib_capture_sequence_async(self, save_path=None):
        """Handle the complete capture sequence asynchronously - simplified using position sync"""
        def calib_capture_worker(position_list):
            capture_thread = None
            capture_threads = []
            try:
                self._logger.debug("Started calibration capture worker")
                
                # # 1. Send movement commands  
                # home_commands = gcd.gen_home_axes_gcode()
                # self._printer.commands(home_commands, tags={'plugin:GcodeQueuingInjection', 'home-axes'})
                
                for position in position_list:
                    move_commands = gcd.gen_move_simple_gcode(
                        position=position,
                        feedrate=self._get_move_feedrate(),
                    )
                    self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                    
                    # 2. Wait for moves to complete using position sync
                    if self.get_position_async() is None:
                        self._logger.error("Failed to reach capture position, aborting")
                        return
                    
                    # 3. Prepare for capture completion signaling
                    self._capture_signal.clear()  # Clear signal before starting capture
                    
                    # 4. Wait for wait_before_capture_ms
                    time.sleep(self._get_wait_before_capture_ms() / 1000)
                    
                    # 5. Capture image - printer is now guaranteed to be in position
                    capture_thread = threading.Thread(
                        target=self.capture_img, args=(position, None, position["z"], save_path))
                    capture_thread.start()
                    capture_threads.append(capture_thread) 
                    
                    # 6. Wait for capture completion signal instead of fixed delay
                    if not self.wait_for_capture_completion():
                        self._logger.error("Capture completion timeout, continuing anyway")
                                    
            
                # 7. Send movement commands  
                home_commands = gcd.gen_home_axes_gcode()
                self._printer.commands(home_commands, tags={'plugin:GcodeQueuingInjection', 'home-axes'})

                self._logger.debug("Calibration capture sequence completed successfully")
    
            except Exception as e:
                self._logger.error("Error in calibration capture sequence: %s\n%s", e, traceback.format_exc())
            finally:
                # Wait for image capture to complete BEFORE releasing job hold
                for cap_thread in capture_threads:
                    cap_thread.join(timeout=10.0)
                    self._logger.debug("Image capture thread completed")
                
                # Always release the job hold LAST
                self._printer.set_job_on_hold(False)
                self._logger.debug("Released job hold lock")

        position_list = calib_capture.get_calib_capture_positions()
        # Start the worker thread
        worker_thread = threading.Thread(target=calib_capture_worker, args=(position_list,))
        worker_thread.daemon = True
        worker_thread.start()



    def move_to_calib_position(self):
        """Handle the complete capture sequence asynchronously - simplified using position sync"""
        def calib_capture_worker(position):
            try:
                self._logger.debug("Started calibration capture worker")
                
                # # 1. Send movement commands  
                # home_commands = gcd.gen_home_axes_gcode()
                # self._printer.commands(home_commands, tags={'plugin:GcodeQueuingInjection', 'home-axes'})
                
                move_commands = gcd.gen_move_simple_gcode(
                    position=position,
                    feedrate=self._get_move_feedrate(),
                )
                self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                
                # 2. Wait for moves to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to reach capture position, aborting")
                    return
            except Exception as e:
                self._logger.error("Error in move to calib position: %s\n%s", e, traceback.format_exc())
            
        position = calib_capture.get_singlecalib_capture_position()
        # Start the worker thread
        worker_thread = threading.Thread(target=calib_capture_worker, args=(position,))
        worker_thread.daemon = True
        worker_thread.start()


    def api_run_capture(self, save_path=None, filename=None):
        """Handle the complete capture sequence asynchronously - simplified using position sync"""
        def calib_capture_worker(position, filename):
            capture_thread = None
            capture_threads = []
            
            self._capture_signal.clear()  # Clear signal before starting capture

            # 1. Wait for wait_before_capture_ms
            time.sleep(self._get_wait_before_capture_ms() / 1000)

            # 2. Capture image - printer is now guaranteed to be in position
            capture_thread = threading.Thread(
                target=self.capture_img, args=(position, None, position["z"], save_path, filename))
            capture_thread.start()
            capture_threads.append(capture_thread) 

            # 3. Wait for capture completion signal instead of fixed delay
            if not self.wait_for_capture_completion():
                self._logger.error("Capture completion timeout, continuing anyway")
            

        position = calib_capture.get_singlecalib_capture_position()
        # Start the worker thread
        worker_thread = threading.Thread(target=calib_capture_worker, args=(position, filename))
        worker_thread.daemon = True
        worker_thread.start()

