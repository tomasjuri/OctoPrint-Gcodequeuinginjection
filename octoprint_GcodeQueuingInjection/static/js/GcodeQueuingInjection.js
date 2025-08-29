/*
 * View model for OctoPrint-GcodeQueuingInjection
 *
 * Author: linux-paul
 * License: AGPLv3
 */
$(function() {
    console.log("GcodeQueuingInjection plugin JS loading...");
    console.log("PLUGIN_BASEURL:", window.PLUGIN_BASEURL);
    
    // Extend the settings view model to add our custom functionality
    OCTOPRINT_VIEWMODELS.push({
        construct: function(parameters) {
            console.log("GcodeQueuingInjection view model constructing...");
            var self = this;
            
            // Get the settings view model
            var settingsViewModel = parameters[0];
            console.log("Settings view model:", settingsViewModel);
        },
        dependencies: ["settingsViewModel"],
        elements: ["#settings_plugin_GcodeQueuingInjection"]
    });
    
    console.log("GcodeQueuingInjection plugin JS loaded");
});
