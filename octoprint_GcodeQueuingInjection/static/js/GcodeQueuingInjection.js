/*
 * View model for OctoPrint-Gcodequeuinginjection
 *
 * Author: Tomas Jurica
 * License: AGPL-3.0-or-later
 */
$(function() {
    function GcodequeuinginjectionViewModel(parameters) {
        var self = this;

        // Assign the injected parameters
        self.settingsViewModel = parameters[0];

        // Test results observables
        self.testResults = ko.observable("");
        self.testSuccess = ko.observable(false);

        // Test camera connection
        self.testCameraConnection = function() {
            self.testResults("");
            self.testSuccess(false);
            
            var url = self.settingsViewModel.settings.plugins.GcodeQueuingInjection.snapshot_url();
            
            if (!url) {
                self.testResults("Please enter a snapshot URL first.");
                self.testSuccess(false);
                return;
            }

            // Test the camera URL with a HEAD request to avoid downloading the image
            $.ajax({
                url: url,
                type: 'HEAD',
                timeout: 5000,
                success: function() {
                    self.testResults("Camera connection successful! URL is responding.");
                    self.testSuccess(true);
                },
                error: function(xhr, status, error) {
                    var message = "Camera connection failed: ";
                    if (status === "timeout") {
                        message += "Connection timeout. Check if the URL is correct and the camera is accessible.";
                    } else if (xhr.status === 0) {
                        message += "Cannot reach the URL. Check if the camera service is running.";
                    } else {
                        message += "HTTP " + xhr.status + " - " + error;
                    }
                    self.testResults(message);
                    self.testSuccess(false);
                }
            });
        };

        // Validate capture folder
        self.validateCaptureFolder = function() {
            self.testResults("");
            self.testSuccess(false);
            
            var folder = self.settingsViewModel.settings.plugins.GcodeQueuingInjection.capture_folder();
            
            if (!folder) {
                self.testResults("Please enter a capture folder path first.");
                self.testSuccess(false);
                return;
            }

            // Send a request to the plugin to validate the folder
            $.ajax({
                url: API_BASEURL + "plugin/GcodeQueuingInjection",
                type: "POST",
                dataType: "json",
                data: JSON.stringify({
                    command: "validate_folder",
                    folder: folder
                }),
                contentType: "application/json; charset=UTF-8",
                success: function(response) {
                    if (response.valid) {
                        self.testResults("Folder is valid and accessible: " + response.resolved_path);
                        self.testSuccess(true);
                    } else {
                        self.testResults("Folder validation failed: " + response.error);
                        self.testSuccess(false);
                    }
                },
                error: function() {
                    self.testResults("Failed to communicate with the plugin for folder validation.");
                    self.testSuccess(false);
                }
            });
        };

        // Clear test results when settings change
        self.clearTestResults = function() {
            self.testResults("");
        };

        // Subscribe to setting changes to clear test results
        self.onBeforeBinding = function() {
            if (self.settingsViewModel && self.settingsViewModel.settings && 
                self.settingsViewModel.settings.plugins && 
                self.settingsViewModel.settings.plugins.GcodeQueuingInjection) {
                self.settingsViewModel.settings.plugins.GcodeQueuingInjection.snapshot_url.subscribe(self.clearTestResults);
                self.settingsViewModel.settings.plugins.GcodeQueuingInjection.capture_folder.subscribe(self.clearTestResults);
            }
        };
    }

    /* view model class, parameters for constructor, container to bind to
     * Please see http://docs.octoprint.org/en/master/plugins/viewmodels.html#registering-custom-viewmodels for more details
     * and a full list of the available options.
     */
    OCTOPRINT_VIEWMODELS.push({
        construct: GcodequeuinginjectionViewModel,
        dependencies: ["settingsViewModel"],
        elements: [],
        name: "GcodeQueuingInjection"
    });
});
