import octoprint.plugin
import logging
import random as rnd
from . import gcode_sequences as gcd
import threading
import time
import re
from .config import *

class GcodequeuinginjectionPlugin(octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.TemplatePlugin
):
    def __init__(self):
        self.print_gcode = False
        self.state = "printing"
        assert self.state in ["printing", "waiting_for_position", "stopped"]
        self._position_event = threading.Event()
        self.position = None
        self.cam_offsets = CAM_EXTRUDER_OFFSETS
        self.rnd_offset_range = RANDOM_OFFSET_RANGE

    def gen_capture_pos(self, capture_position, offsets, rnd_range):
        capture_position = {
            "x": capture_position["x"] + offsets["x"] + rnd.uniform(rnd_range["x"][0], rnd_range["x"][1]),
            "y": capture_position["y"] + offsets["y"] + rnd.uniform(rnd_range["y"][0], rnd_range["y"][1]),
            "z": capture_position["z"] + offsets["z"] + rnd.uniform(rnd_range["z"][0], rnd_range["z"][1]),
        }
        return capture_position

    def capture_img(self, capture_position):
        self._logger.debug("Capturing image at position: {capture_position}".format(capture_position=capture_position))
        # TODO: Implement capture
        time.sleep(1)
        self._logger.debug("Capture complete")
        pass

    def gcode_queuing(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        if gcode and gcode == gcd.CAPTURE_GCODE[0]:
            self.state = [c for c in cmd.split(" ") if c.startswith("S")][0]
            
            # S1: ask for position
            if self.state == "S0": 
                self._logger.debug("STATE: printing ==> waiting for position")
                self._position_event.clear()
                cmd = gcd.WAITING_FOR_POS_GCODE
            
            # S2: wait for position and move to capture position
            elif self.state == "S1":
                self._logger.debug("STATE: waiting for position ==> moving_to_capture")
                self._logger.debug("Waiting for position...")
                self._position_event.wait(30)
                self._logger.debug("Position received: {position}".format(position=self.position))
                self.capture_pos = self.gen_capture_pos(
                    self.position, self.cam_offsets, self.rnd_offset_range)
                cmd = gcd.gen_move_to_capture_gcode(
                    capture_position=self.capture_pos,
                    retraction_mm=RETRACTION_MM,
                    retraction_speed=RETRACTION_SPEED,
                )

            # S3: Do the actual capture and return
            elif self.state == "S2":
                self._logger.debug("STATE: moving_to_capture ==> capture_and_return")
                
                capture_thread = threading.Thread(
                    target=self.capture_img, args=(self.capture_pos,))
                capture_thread.start()
                
                cmd = gcd.gen_capture_and_return_gcode(
                    return_position=self.position,
                    retraction_mm=RETRACTION_MM,
                    retraction_speed=RETRACTION_SPEED,
                )
            
            self._logger.debug("Rewriting to: {cmd}".format(**locals()))
            return cmd

    
    def gcode_sent(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        self._logger.debug("Gcode sent: {cmd}, {gcode}".format(**locals()))

        if cmd and cmd == gcd.START_PRINT_GCODE[0]:
            self.print_gcode = True
        if cmd and cmd == gcd.STOP_PRINT_GCODE[0]:
            self.print_gcode = False
        
        if self.print_gcode:
            self._logger.debug("Gcode sent: {cmd}".format(**locals()))

    def gcode_received(self, comm_instance, line, *args, **kwargs):
        """Handle M114 position responses and trigger state transitions"""
        
        position = {"x": None, "y": None, "z": None, "e": None}

        pos_re = r'^ok X:(\d+\.\d+) Y:(\d+\.\d+) Z:(\d+\.\d+) E:(\d+\.\d+) Count: A:'
        pos_matched = re.search(pos_re, line)
        if pos_matched:
            position["x"] = float(pos_matched.group(1))
            position["y"] = float(pos_matched.group(2))
            position["z"] = float(pos_matched.group(3))
            position["e"] = float(pos_matched.group(4))
            self._logger.debug(f"Position received: X: {position['x']}, Y: {position['y']}, Z: {position['z']}, E: {position['e']}")

            self.position = position
            self._position_event.set()

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
