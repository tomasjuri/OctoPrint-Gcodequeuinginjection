

def get_calib_capture_positions():
    """
    Generate the capture positions for the calibration.
    """
    # x_pos = [50, 100, 150, 180, 200]
    # y_pos = [50, 100, 150, 180, 200]
    # z_pos = [150, 180, 210]

    x_pos = [50, 80, 100, 110, 120, 150]
    y_pos = [20, 50, 100, 150, 180, 200]
    z_pos = [150, 200]
    
    position_list = []
    for z in z_pos:
        for y in y_pos:
            for x in x_pos:
                position_list.append({
                    "x": x,
                    "y": y,
                    "z": z,
                })
    return position_list

def get_singlecalib_capture_position():
    """
    Generate the capture positions for the calibration.
    """
    position = {
        "x": 100,
        "y": 100,
        "z": 150,
    }
    return position