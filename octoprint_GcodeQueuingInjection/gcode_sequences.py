

CAPTURE_GCODE = ("M240",)
START_PRINT_GCODE = ("@PRINT",)
STOP_PRINT_GCODE = ("@STOPPRINT",)

WAITING_FOR_POS_GCODE = [
    START_PRINT_GCODE,
    ("M400",), # Wait for all previous commands to complete
    ( "M114",), # Get current location
    STOP_PRINT_GCODE,
    CAPTURE_GCODE, # Trigger next action with this custom gcode
]

MOVE_TO_CAPTURE_GCODE = [
    START_PRINT_GCODE,
    ("M400",), # Wait for all previous commands to complete
    ("G0", "X", "Y", "Z"), # Move to capture position
    ("M400",), # Wait for all previous commands to complete
    STOP_PRINT_GCODE,
    CAPTURE_GCODE, # Trigger next action with this custom gcode
]








