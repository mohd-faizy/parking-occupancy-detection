"""
Parking Space Counter - Main Entry Point
========================================
Run the live parking space detector or the interactive space picker tool.

Usage:
    # Run the live detector with default video
    python main.py

    # Launch the interactive parking slot annotation tool
    python main.py --picker

    # Run with custom video and positions file
    python main.py --video path/to/video.mp4 --pos path/to/CarParkPos
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counter_logic import run_parking_counter, get_default_paths
from src.ParkingSpacePicker import run_picker


def parse_arguments():
    """Parse user command-line arguments."""
    default_vid, default_pos = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Smart Parking Space Occupancy Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "--picker",
        action="store_true",
        help="Launch the interactive parking space position picker tool instead of video counter.",)

    parser.add_argument(
        "--video",
        type=str,
        default=str(default_vid),
        help="Path to the parking lot video file.",)

    parser.add_argument(
        "--pos",
        type=str,
        default=str(default_pos),
        help="Path to the saved parking space coordinates (pickle file).",)
        
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.picker:
        print("[INFO] Launching Parking Space Calibration Tool...")
        run_picker(pos_path=args.pos)
    else:
        print("[INFO] Launching Parking Space Counter Live Demo...")
        run_parking_counter(video_path=args.video, pos_path=args.pos)


if __name__ == "__main__":
    main()

