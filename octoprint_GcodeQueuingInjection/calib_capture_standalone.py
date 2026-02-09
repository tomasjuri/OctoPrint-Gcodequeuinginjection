#!/usr/bin/env python3
"""
Simple chessboard calibration image capture script.
Fetches images from camera service HTTP endpoint - same as OctoPrint addon.

This ensures identical camera settings between calibration captures and
production captures from the OctoPrint plugin.

Usage:
    python calib_capture_standalone.py
    
Controls:
    ENTER - Capture image
    q     - Quit
"""
import requests
from datetime import datetime
import os

# Configuration - uses same endpoint as OctoPrint addon (see config.py)
SNAPSHOT_URL = "http://127.0.0.1:8080/?action=snapshot"
OUTPUT_DIR = os.path.expanduser("~/data/calibration_captures")


def capture_image(url=SNAPSHOT_URL, timeout=10):
    """Fetch snapshot from camera service"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    
    count = 0
    print(f"Capturing from: {SNAPSHOT_URL}")
    print(f"Saving to: {session_dir}")
    print("\nPress ENTER to capture, 'q' to quit\n")
    
    try:
        while True:
            cmd = input(f"[{count} captured] > ").strip().lower()
            if cmd == 'q':
                break
            count += 1
            filename = f"calib_{count:03d}.jpg"
            path = os.path.join(session_dir, filename)
            with open(path, 'wb') as f:
                f.write(capture_image())
            print(f"  Saved: {filename}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    print(f"\nCaptured {count} images to {session_dir}")


if __name__ == "__main__":
    main()
