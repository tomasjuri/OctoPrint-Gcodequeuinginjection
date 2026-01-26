import octoprint.plugin
import logging
import random as rnd
from . import gcode_sequences as gcd
from . import calib_capture
import threading
import time
import numpy as np

import re
from .config import (
    CAM_EXTRUDER_OFFSETS, 
    RANDOM_OFFSET_RANGE, 
    RETRACTION_MM, 
    RETRACTION_SPEED, 
    MOVE_FEEDRATE,
    WAIT_BEFORE_CAPTURE_MS,
    SNAPSHOT_URL,
    CAPTURE_FOLDER
)
from .camera import Camera
from datetime import datetime

import json
import os

class GcodequeuinginjectionPlugin(
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.SimpleApiPlugin
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

    def gen_capture_pos(self, cmd, capture_position, offsets, rnd_range):
        """Generate capture position with offsets and random variations."""
        # I start the capture squence with M240 Z<height> ZN<layer_num> S<state> Gcode
        # Example: M240 Z0.4 ZN1 S0
        parts = cmd.split(" ")
        
        # Extract Z height (remove 'Z' prefix)
        layer_height = float(parts[1][1:])  # parts[1] = "Z0.4" -> "0.4"
        # Extract layer number (remove 'ZN' prefix)
        layer_n = int(parts[2][2:])     # parts[2] = "ZN1" -> "1"
        
        capture_position = {
            "x": capture_position["x"] + offsets["x"] + rnd.uniform(rnd_range["x"][0], rnd_range["x"][1]),
            "y": capture_position["y"] + offsets["y"] + rnd.uniform(rnd_range["y"][0], rnd_range["y"][1]),
            "z": capture_position["z"] + offsets["z"] + rnd.uniform(rnd_range["z"][0], rnd_range["z"][1]),
        }
        return capture_position, layer_n, layer_height

    def parse_position_line(self, line):
        """Parse M114 response"""
        regex_float_pattern = r"[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?"
        regex_position = re.compile(
            r"X:\s*(?P<x>{float})\s*Y:\s*(?P<y>{float})\s*Z:\s*(?P<z>{float})\s*(?:E:\s*(?P<e>{float}))?"
            .format(float=regex_float_pattern))
        
        match = regex_position.search(line)
        if match is not None:
            result = {
                'x': float(match.group("x")),
                'y': float(match.group("y")),
                'z': float(match.group("z")),
                'e': float(match.group("e"))
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
        

    def capture_sequence_async(self, original_cmd):
        """Handle the complete capture sequence asynchronously - simplified using position sync"""
        def capture_worker():
            capture_thread = None
            try:
                self._logger.debug("Started async capture worker")
                
                # Get initial position - this also serves as sync point
                start_position = self.get_position_async()
                if start_position is None:
                    self._logger.error("Failed to get position, aborting capture")
                    return
                
                # Generate capture position
                capture_pos, layer_n, layer_height = self.gen_capture_pos(
                    original_cmd, start_position, self._settings.get(
                        ["cam_extruder_offsets"]), self._settings.get(["random_offset_range"]))
                
                # 1. Send movement commands  
                move_commands = gcd.gen_move_to_capture_gcode(
                    capture_position=capture_pos,
                    retraction_mm=self._settings.get(["retraction_mm"]),
                    retraction_speed=self._settings.get(["retraction_speed"]),
                )
                self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                
                # 2. Wait for moves to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to reach capture position, aborting")
                    return
                
                # 3. Prepare for capture completion signaling
                self._capture_signal.clear()  # Clear signal before starting capture
                
                # 4. Wait for wait_before_capture_ms
                self._logger.debug(f"Waiting for {self._settings.get(['wait_before_capture_ms'])} ms before capture")
                time.sleep(self._settings.get(["wait_before_capture_ms"]) / 1000)
                
                # 5. Capture image - printer is now guaranteed to be in position
                capture_thread = threading.Thread(
                    target=self.capture_img, args=(capture_pos, layer_n, layer_height))
                capture_thread.start()
                
                # 6. Wait for capture completion signal instead of fixed delay
                if not self.wait_for_capture_completion():
                    self._logger.error("Capture completion timeout, continuing anyway")
                                    
                # 7. Send return commands after capture is confirmed complete
                return_commands = gcd.gen_capture_and_return_gcode(
                    return_position=start_position,
                    retraction_mm=self._settings.get(["retraction_mm"]),
                    retraction_speed=self._settings.get(["retraction_speed"]),
                )
                self._printer.commands(return_commands, tags={'plugin:GcodeQueuingInjection', 'capture-return'})
                
                # 8. Wait for return sequence to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to complete return sequence")
                
                self._logger.debug("Capture sequence completed successfully")
                    
            except Exception as e:
                self._logger.error("Error in capture sequence: %s", e)
            finally:
                # Wait for image capture to complete BEFORE releasing job hold
                if capture_thread:
                    capture_thread.join(timeout=10.0)
                    self._logger.debug("Image capture thread completed")
                
                # Always release the job hold LAST
                self._printer.set_job_on_hold(False)
                self._logger.debug("Released job hold lock")

        
        # Start the worker thread
        worker_thread = threading.Thread(target=capture_worker)
        worker_thread.daemon = True
        worker_thread.start()

    def gcode_queuing(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        """Handle gcode queuing."""
        if gcode and gcode == gcd.CAPTURE_GCODE[0]:
            self.state = [c for c in cmd.split(" ") if c.startswith("S")][0]
            
            # S0: Start capture sequence using Octolapse pattern
            if self.state == "S0": 
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
                        feedrate=self._settings.get(["move_feedrate"]),
                    )
                    self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                    
                    # 2. Wait for moves to complete using position sync
                    if self.get_position_async() is None:
                        self._logger.error("Failed to reach capture position, aborting")
                        return
                    
                    # 3. Prepare for capture completion signaling
                    self._capture_signal.clear()  # Clear signal before starting capture
                    
                    # 4. Wait for wait_before_capture_ms
                    time.sleep(self._settings.get(["wait_before_capture_ms"]) / 1000)
                    
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
                self._logger.error("Error in capture sequence: %s", e)
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
                    feedrate=self._settings.get(["move_feedrate"]),
                )
                self._printer.commands(move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                
                # 2. Wait for moves to complete using position sync
                if self.get_position_async() is None:
                    self._logger.error("Failed to reach capture position, aborting")
                    return
            except Exception as e:
                self._logger.error("Error in capture sequence: %s", e)
            
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
            time.sleep(self._settings.get(["wait_before_capture_ms"]) / 1000)

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

