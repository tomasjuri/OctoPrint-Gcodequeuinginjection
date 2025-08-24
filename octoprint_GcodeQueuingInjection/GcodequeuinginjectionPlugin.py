import octoprint.plugin
import logging
import random as rnd
from . import gcode_sequences as gcd
import threading
import time
import re
from .config import *
from .camera import Camera
from datetime import datetime
import PIL
import json
import os

class GcodequeuinginjectionPlugin(octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.TemplatePlugin
):
    def __init__(self):
        self.print_gcode = False

        self._logger = logging.getLogger(__name__)
        
        # Octolapse-style position tracking
        self._position_signal = threading.Event()
        self._position_signal.set()  # Start in set state
        self._position_request_sent = False
        self._position_payload = None
        self._position_timeout = 30.0
        
        self.cam_offsets = CAM_EXTRUDER_OFFSETS
        self.rnd_offset_range = RANDOM_OFFSET_RANGE

        self.camera = Camera()
        self.init_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.capture_log_file_path = os.path.join(self._get_save_path(), "gcode_capture.log")


    def _get_save_path(self):
        """Get the configured save path"""
        save_path = os.path.expanduser(CAPTURE_FOLDER)
        save_path = os.path.join(save_path, self.init_timestamp)
        if not os.path.exists(save_path):
            self._logger.debug("Creating capture folder: {}".format(save_path))
            os.makedirs(save_path)
        return save_path

    def gen_capture_pos(self, cmd, capture_position, offsets, rnd_range):
        # Parse M240 Z<height> ZN<layer_num> S<state> format
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
                'z': float(match.group("z"))
            }
            if match.group("e") is not None:
                result["e"] = float(match.group("e"))
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

    def on_position_received(self, payload):
        """Handle position response"""
        if self._position_request_sent:
            self._position_request_sent = False
            self._logger.debug("Position received: {}".format(payload))
            self._position_payload = payload
            self._position_signal.set()
        else:
            self._logger.debug("Position response received but not requested by us, ignoring")

    def capture_img(self, capture_position, layer_n, layer_height):
        self._logger.debug("Capturing image. Expected to be at position: {capture_position}".format(capture_position=capture_position))
        
        img = self.camera.capture_image()
        self._logger.debug("Image captured")
        
        img_filepath = f"img_{layer_n:03d}.jpg"
        json_filepath = f"img_{layer_n:03d}.json"

        im_path = os.path.join(self._get_save_path(), img_filepath)
        img.save(im_path)

        data = {
            "img_filepath": img_filepath,
            "json_filepath": json_filepath,
            "capture_position": capture_position,
            "layer_n": layer_n,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "layer_height": layer_height,
            "calibration_data": "TODO",
        }
        json_path = os.path.join(self._get_save_path(), json_filepath)
        with open(json_path, "w") as f:
            json.dump(data, f)

        self._logger.debug("Capture complete, saved to {im_path} and {json_path}".format(**locals()))
        

    def capture_sequence_async(self, original_cmd):
        """Handle the complete capture sequence asynchronously using Octolapse pattern"""
        def capture_worker():
            try:
                self._logger.debug("Started async capture worker")
                
                # Request position using Octolapse pattern
                position = self.get_position_async()
                
                if position is None:
                    self._logger.error("Failed to get position, aborting capture")
                    return
                
                # Generate capture position
                try:
                    capture_pos, layer_n, layer_height = self.gen_capture_pos(
                        original_cmd, position, self.cam_offsets, self.rnd_offset_range)
                    
                    # Generate and send movement commands  
                    move_commands = gcd.gen_move_to_capture_gcode(
                        capture_position=capture_pos,
                        retraction_mm=RETRACTION_MM,
                        retraction_speed=RETRACTION_SPEED,
                    )
                    self._logger.debug("Sending move to capture commands: {}".format(move_commands))
                    
                    # Send movement commands first
                    self._printer.commands(
                        move_commands, tags={'plugin:GcodeQueuingInjection', 'capture-move'})
                    
                    # Wait for movement commands to complete using separate position request
                    final_position = self.get_position_async()
                    
                    if final_position is None:
                        self._logger.error("Failed to reach capture position, aborting capture")
                        return
                    
                    self._logger.debug("Printer reached capture position: {}".format(final_position))
                    
                    # NOW capture image - printer is guaranteed to be in position
                    capture_thread = threading.Thread(
                        target=self.capture_img, args=(capture_pos, layer_n, layer_height))
                    capture_thread.start()
                                        
                    # Send return commands
                    return_commands = gcd.gen_capture_and_return_gcode(
                        return_position=position,
                        retraction_mm=RETRACTION_MM,
                        retraction_speed=RETRACTION_SPEED,
                    )
                    self._printer.commands(
                        return_commands, tags={'plugin:GcodeQueuingInjection', 'capture-return'})
                    self._logger.debug("Sent return commands")
                    
                except Exception as e:
                    self._logger.error(f"Failed to execute capture sequence: {e}")
                    
            except Exception as e:
                self._logger.error(f"Error in capture sequence: {e}")
            finally:
                # Always release the job hold
                self._printer.set_job_on_hold(False)
                self._logger.debug("Released job hold lock")
                
                # Wait for image capture to complete before returning
                capture_thread.join(timeout=10.0)  # 10 second timeout for image capture

        
        # Start the worker thread
        worker_thread = threading.Thread(target=capture_worker)
        worker_thread.daemon = True
        worker_thread.start()

    def gcode_queuing(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        if gcode and gcode == gcd.CAPTURE_GCODE[0]:
            self.state = [c for c in cmd.split(" ") if c.startswith("S")][0]
            
            # S0: Start capture sequence using Octolapse pattern
            if self.state == "S0": 
                self._logger.debug("S0 STATE: Starting capture sequence with command: {cmd}".format(**locals()))
                
                # Use job_on_hold to pause queue processing
                if self._printer.set_job_on_hold(True):
                    self._logger.debug("Job on hold acquired, starting the sequence")
                    self.capture_sequence_async(cmd)
                    return None,
                else:
                    self._logger.warning("Could not set job on hold, falling back")
                    return None,
            
            self._logger.debug("Rewriting to: {cmd}".format(**locals()))
            return cmd

    
    def gcode_sent(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        if cmd and cmd == gcd.START_PRINT_GCODE[0]:
            self.print_gcode = True
        if cmd and cmd == gcd.STOP_PRINT_GCODE[0]:
            self.print_gcode = False
        
        if self.print_gcode:
            self._logger.debug("Gcode sent: {cmd}".format(**locals()))
            
        with open(self.capture_log_file_path, "a") as f:
            f.write(f"{cmd}\n")


    def gcode_received(self, comm_instance, line, *args, **kwargs):
        """Handle M114 position responses using Octolapse pattern"""
        
        # Only process if we requested the position (Octolapse pattern)
        if self._position_request_sent and 'X:' in line and 'Y:' in line and 'Z:' in line:
            self._logger.debug("Processing position response: {}".format(line))
            
            position = self.parse_position_line(line)
            if position:
                self.on_position_received(position)
            else:
                self._logger.warning("Failed to parse position from line: {}".format(line))
        
        return line

    def get_assets(self):
        # Define your plugin's asset files to automatically include in the
        # core UI here.
        return {
            "js": ["js/GcodeQueuingInjection.js"],
            "css": ["css/GcodeQueuingInjection.css"],
            "less": ["less/GcodeQueuingInjection.less"]
        }

    ##~~ Softwareupdate hook

    def get_update_information(self):
        # Define the configuration for your plugin to use with the Software Update
        # Plugin here. See https://docs.octoprint.org/en/master/bundledplugins/softwareupdate.html
        # for details.
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
