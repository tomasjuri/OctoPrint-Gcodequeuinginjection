from .config import *

CAPTURE_GCODE = ("M240",)
START_PRINT_GCODE = ("@PRINT",)
STOP_PRINT_GCODE = ("@STOPPRINT",)

WAITING_FOR_POS_GCODE = [
    "@WAITING_FOR_POS",
    "M400",  # Wait for all previous commands to complete
    "M114",  # Get current location
    "@WAITING_FOR_POS_END",
]

def gen_move_to_capture_gcode(capture_position, retraction_mm, retraction_speed):
    cmd = [
        "@MOVE_TO_CAPTURE",
        "M83",   # Relative extruder mode
        f"G1 E-{retraction_mm} F{retraction_speed}",  # Retract
        
        "G90",   # Set to absolute mode
        f"G0 X{capture_position['x']} Y{capture_position['y']} Z{capture_position['z']}",  # Move to capture position
        "@MOVE_TO_CAPTURE_END",
    ]
    return cmd
        
def gen_capture_and_return_gcode(return_position, retraction_mm, retraction_speed):
    original_e_pos = return_position.get("e")
    print(f"generating command to return to: {return_position}")
    
    cmd = [
        "@CAPTURE_AND_RETURN",
        f"G4 P{CAPTURE_WAIT_TIME_MS}",  # Wait for capture to complete

        "G90",   # Set to absolute mode
        f"G0 X{return_position['x']} Y{return_position['y']} Z{return_position['z']}",  # Move back to print position
        
        # Undo the retraction first (relative extrusion)
        "M83",   # Set to relative extruder mode
        f"G1 E{retraction_mm} F{retraction_speed}",  # Undo retraction (push filament back)
        
        "M82",   # Set to absolute extruder mode
        f"G92 E{original_e_pos}",
        "@CAPTURE_AND_RETURN_END",
    ]
    
    return cmd
        







# ; 1. INITIALIZATION (if needed)
# M400                    ; Wait for moves to finish

# ; 2. START GCODE - Preparation
# M83                     ; Set extruder to relative mode (if needed)
# G1 E-1.000 F1800       ; Retract filament (if retraction enabled)
# G91                     ; Set to relative mode (if needed) 
# G1 Z2.000 F300         ; Z-lift (if Z-hop enabled)

# ; 3. SNAPSHOT COMMANDS - Move to position
# G90                     ; Set to absolute mode (if needed)
# G0 X100.000 Y100.000 F6000  ; Travel to snapshot position

# ; 4. SNAPSHOT TRIGGER
# @OCTOLAPSE TAKE-SNAPSHOT     ; Trigger snapshot capture

# ; 5. RETURN COMMANDS - Go back
# G0 X50.000 Y50.000 F6000     ; Return to original position

# ; 6. END GCODE - Restore state  
# G1 Z-2.000 F300         ; Lower Z back down (if was lifted)
# G1 E1.000 F1800         ; De-retract filament (if was retracted)
# G90                     ; Restore original coordinate mode
# M82                     ; Restore original extruder mode
# G1 F1200                ; Restore original feedrate








