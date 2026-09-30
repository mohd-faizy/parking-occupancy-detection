"""
Parking Space Counter Package
=============================
Computer Vision toolkit for parking space occupancy detection.
"""

from .counter_logic import (
    load_parking_positions,
    preprocess_image,
    check_parking_spaces,
    draw_dashboard_banner,
    render_dashboard_sidebar,
    run_parking_counter,
    SLOT_WIDTH,
    SLOT_HEIGHT,
    PIXEL_THRESHOLD,
)
from .ParkingSpacePicker import run_picker

__all__ = [
    "load_parking_positions",
    "preprocess_image",
    "check_parking_spaces",
    "draw_dashboard_banner",
    "render_dashboard_sidebar",
    "run_parking_counter",
    "run_picker",
    "SLOT_WIDTH",
    "SLOT_HEIGHT",
    "PIXEL_THRESHOLD",
]

