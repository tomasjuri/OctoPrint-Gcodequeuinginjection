import octoprint.plugin
import logging


class GcodequeuinginjectionPlugin(octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.TemplatePlugin
):
    def __init__(self):
        self.print_gcode = False

    def gcode_queuing(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        if gcode and gcode == "M240":
            self._logger.debug("Queuing M240: {cmd}".format(**locals()))
            cmd = [
                ("G-PRINT",),
                ("M106 S0",),
                ("G-STOPPRINT",),
                ("M106 S0",),
            ]
            self._logger.debug("Rewriting to: {cmd}".format(**locals()))
        return cmd

    def gcode_sent(self, comm_instance, phase, cmd, cmd_type, gcode, *args, **kwargs):
        self._logger.debug("Gcode sent: {cmd}, {gcode}".format(**locals()))

        if cmd and cmd == "G-PRINT":
            self.print_gcode = True
        if cmd and cmd == "G-STOPPRINT":
            self.print_gcode = False
        
        if self.print_gcode:
            self._logger.debug("Gcode sent: {cmd}".format(**locals()))

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
