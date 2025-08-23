

CAPTURE_GCODE = ("M240",)
START_PRINT_GCODE = ("@PRINT",)
STOP_PRINT_GCODE = ("@STOPPRINT",)

WAITING_FOR_POS_GCODE = [
    START_PRINT_GCODE,
    ("M400",), # Wait for all previous commands to complete
    ("M114",), # Get current location
    STOP_PRINT_GCODE,
]

def gen_move_to_capture_gcode(capture_position, retraction_mm, retraction_speed):
    cmd = [
        START_PRINT_GCODE,
        ("M83",),   # Relative extruder mode
        ("G1 E-{retraction_mm} F{retraction_speed}",),  # Retract
        ("M82",), # absolute extruder mode
        ("M400",),  # Wait for retraction
        
        ("G90",),            # Set to absolute mode (if needed)
        ("G0", "X{capture_position['x']}", "Y{capture_position['y']}", "Z{capture_position['z']}"), # Move to capture position
        ("M400",), # Wait for all previous commands to complete

        ("G4 P300",), # Wait for 300ms
        STOP_PRINT_GCODE,
    ]
    return cmd
        
def gen_capture_and_return_gcode(return_position, retraction_mm, retraction_speed):
    move_pos = {
        "x": return_position["x"],
        "y": return_position["y"],
        "z": return_position["z"],
    }
    cmd = [
        START_PRINT_GCODE,
        ("G4 P800",), # Wait for 800ms for capture to complete

        ("G90",),            # Set to absolute mode (if needed)
        ("G0", "X{move_pos['x']}", "Y{move_pos['y']}", "Z{move_pos['z']}"), # Move to capture position
        ("M400",), # Wait for all previous commands to complete
        ("M83",),   # Relative extruder mode
        ("G1 E-{retraction_mm} F{retraction_speed}",),  # Retract
        ("M400",),  # Wait for retraction
        STOP_PRINT_GCODE,
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








