"""
Parking Space Counter Logic
===========================
This module provides functions to process video frames and classify parking
spaces as Free or Occupied based on edge density and pixel thresholding.

Core Workflow:
1. Load pre-annotated parking space positions from file.
2. Preprocess video frames (Grayscale -> Gaussian Blur -> Adaptive Threshold -> Median Blur -> Dilation).
3. Crop each parking slot from the processed image.
4. Count white pixels using cv2.countNonZero().
5. If count < threshold: Space is FREE (Green).
   If count >= threshold: Space is OCCUPIED (Red).
"""

from pathlib import Path
import pickle
import cv2
import numpy as np

# Standard parking space bounding box dimensions (in pixels)
SLOT_WIDTH = 107
SLOT_HEIGHT = 48

# Default non-zero pixel threshold to distinguish between empty asphalt and a car
PIXEL_THRESHOLD = 900


def get_default_paths():
    """
    Resolve default paths for assets safely across different working directories.
    
    Returns:
        tuple[Path, Path]: (video_path, positions_path)
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # Try asset folder in project root first, then inside src
    candidate_video_paths = [
        project_root / "asset" / "carPark.mp4",
        current_dir / "asset" / "carPark.mp4",
        Path("asset/carPark.mp4")
    ]
    candidate_pos_paths = [
        project_root / "asset" / "CarParkPos",
        current_dir / "asset" / "CarParkPos",
        Path("asset/CarParkPos")
    ]

    video_path = next((p for p in candidate_video_paths if p.exists()), candidate_video_paths[0])
    pos_path = next((p for p in candidate_pos_paths if p.exists()), candidate_pos_paths[0])

    return video_path, pos_path


def load_parking_positions(pos_path=None):
    """
    Load saved parking space coordinates from a pickle file.
    
    Args:
        pos_path (str or Path, optional): Path to the CarParkPos pickle file.
        
    Returns:
        list of tuple[int, int]: List of top-left (x, y) coordinates for each spot.
    """
    if pos_path is None:
        _, pos_path = get_default_paths()
    
    pos_file = Path(pos_path)
    if not pos_file.exists():
        print(f"[WARNING] Positions file not found at: {pos_file}")
        return []

    try:
        with open(pos_file, "rb") as file:
            positions = pickle.load(file)
            print(f"[INFO] Loaded {len(positions)} parking space coordinates from {pos_file.name}")
            return positions
    except Exception as error:
        print(f"[ERROR] Failed to load positions file: {error}")
        return []


def preprocess_image(frame):
    """
    Preprocess a raw BGR camera/video frame into a clean binary edge map.
    
    Steps:
    1. Grayscale: Simplifies 3 color channels (BGR) to 1 intensity channel.
    2. Gaussian Blur: Smooths minor sensor noise and texture variations.
    3. Adaptive Threshold: Converts image to binary based on local neighborhood contrast.
    4. Median Blur: Cleans salt-and-pepper noise while preserving sharp boundaries.
    5. Morphological Dilation: Thickens edges so car features merge into solid regions.
    
    Args:
        frame (np.ndarray): Original BGR input image.
        
    Returns:
        np.ndarray: Single-channel binary image ready for pixel counting.
    """
    # Step 1: Convert BGR image to single-channel Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to reduce high-frequency camera noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)

    # Step 3: Adaptive thresholding handles non-uniform lighting and shadows
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        16
    )

    # Step 4: Median blur removes salt-and-pepper noise
    median = cv2.medianBlur(thresholded, 5)

    # Step 5: Dilate to close small gaps in car edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)

    return dilated


def check_parking_spaces(img_processed, img_display, pos_list,
                         width=SLOT_WIDTH, height=SLOT_HEIGHT,
                         threshold=PIXEL_THRESHOLD, show_labels=True):
    """
    Evaluate each parking spot, count non-zero pixels, and draw color-coded rectangles.
    
    Args:
        img_processed (np.ndarray): Binary dilated image for pixel analysis.
        img_display (np.ndarray): Original BGR image to draw rectangles and labels onto.
        pos_list (list): List of (x, y) top-left coordinates for each parking slot.
        width (int): Width of parking spot rectangle.
        height (int): Height of parking spot rectangle.
        threshold (int): Maximum white pixel count allowed for a slot to be marked FREE.
        show_labels (bool): Whether to overlay the numeric pixel count inside each slot.
        
    Returns:
        tuple[int, int, list]: (free_spaces_count, total_spaces_count, slot_details)
    """
    free_spaces = 0
    slot_details = []

    for index, (x, y) in enumerate(pos_list):
        # Crop the bounding box of the parking slot from the binary processed image
        slot_crop = img_processed[y : y + height, x : x + width]
        
        # Count white (non-zero) pixels inside the crop
        count = cv2.countNonZero(slot_crop)
        is_free = count < threshold

        if is_free:
            # Vibrant Green outline for FREE spot
            color = (0, 230, 0)
            thickness = 3
            free_spaces += 1
        else:
            # Clean Red outline for OCCUPIED spot
            color = (0, 0, 230)
            thickness = 2

        # Draw slot bounding box
        cv2.rectangle(img_display, (x, y), (x + width, y + height), color, thickness)
        
        # Optionally draw clean pixel count text inside slot
        if show_labels:
            cv2.putText(
                img_display,
                str(count),
                (x + 4, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        slot_details.append({
            "slot_id": index + 1,
            "x": x,
            "y": y,
            "count": count,
            "is_free": is_free
        })

    return free_spaces, len(pos_list), slot_details


def draw_dashboard_banner(img, free_count, total_count):
    """
    Legacy compact status banner on top-left corner (kept for standalone use).
    
    Args:
        img (np.ndarray): Target BGR image.
        free_count (int): Number of available parking slots.
        total_count (int): Total number of designated parking slots.
    """
    occupied_count = total_count - free_count
    cv2.rectangle(img, (15, 15), (320, 85), (30, 30, 30), cv2.FILLED)
    cv2.rectangle(img, (15, 15), (320, 85), (0, 200, 0), 2)
    status_text = f"FREE: {free_count}/{total_count}"
    cv2.putText(img, status_text, (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    detail_text = f"Occupied: {occupied_count}"
    cv2.putText(img, detail_text, (32, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def render_dashboard_sidebar(height, free_spaces, total_spaces, slot_details, history_trend=None, width=380):
    """
    Render a high-tech, modern visual analytics sidebar with donut chart, KPI cards,
    zone/aisle breakdown, and real-time vacancy trend graph.
    
    Args:
        height (int): Height of the video frame in pixels (e.g. 720).
        free_spaces (int): Count of currently vacant spots.
        total_spaces (int): Total designated parking capacity.
        slot_details (list): Detailed list of per-slot classification dictionaries.
        history_trend (list or deque, optional): Recent history of free space counts.
        width (int): Sidebar panel width in pixels (default 380).
        
    Returns:
        np.ndarray: BGR sidebar canvas of dimensions (height, width, 3).
    """
    sidebar = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient dark slate background
    for row in range(height):
        factor = row / height
        b = int(22 + 8 * factor)
        g = int(20 + 6 * factor)
        r = int(24 + 6 * factor)
        sidebar[row, :] = (b, g, r)

    # Subtle vertical divider line on right edge
    sidebar[:, -2:] = (50, 50, 60)

    occupied_spaces = total_spaces - free_spaces
    occ_pct = int(round((occupied_spaces / total_spaces) * 100)) if total_spaces > 0 else 0
    free_pct = 100 - occ_pct

    # Helper function to draw rounded cards
    def draw_card(canvas, x1, y1, x2, y2, bg_color=(32, 28, 36), border_color=(60, 55, 70)):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    # 1. Header Card
    draw_card(sidebar, 15, 14, width - 15, 74, bg_color=(30, 26, 34), border_color=(70, 65, 80))
    cv2.putText(sidebar, 'PARKING RADAR AI', (28, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    # Glowing green live pulse indicator
    cv2.circle(sidebar, (30, 59), 5, (0, 230, 0), cv2.FILLED)
    cv2.putText(sidebar, 'LIVE TELEMETRY  |  24 FPS', (43, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Donut Gauge Chart Card
    draw_card(sidebar, 15, 86, width - 15, 246, bg_color=(30, 26, 34), border_color=(70, 65, 80))
    cv2.putText(sidebar, 'OCCUPANCY GAUGE', (28, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    center_x, center_y = 105, 158
    radius, thick = 42, 10

    # Background ring track
    cv2.circle(sidebar, (center_x, center_y), radius, (45, 40, 50), thick, cv2.LINE_AA)

    # Occupied arc (Red) & Free arc (Green)
    occ_angle = int((occupied_spaces / total_spaces) * 360) if total_spaces > 0 else 0
    if occ_angle > 0:
        cv2.ellipse(sidebar, (center_x, center_y), (radius, radius), -90, 0, occ_angle, (30, 30, 220), thick, cv2.LINE_AA)
    if occ_angle < 360:
        cv2.ellipse(sidebar, (center_x, center_y), (radius, radius), -90, occ_angle, 360, (30, 210, 50), thick, cv2.LINE_AA)

    # Donut center text
    cv2.putText(sidebar, f'{occ_pct}%', (center_x - 22, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sidebar, 'FULL', (center_x - 14, center_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)

    # Load Status Text next to gauge
    load_status = 'High Load' if occ_pct > 75 else ('Moderate' if occ_pct > 40 else 'Available')
    status_color = (60, 60, 240) if occ_pct > 75 else ((60, 200, 240) if occ_pct > 40 else (50, 230, 80))
    cv2.putText(sidebar, f'Status: {load_status}', (175, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 1, cv2.LINE_AA)
    # Legend bullets
    cv2.circle(sidebar, (180, 158), 5, (30, 210, 50), cv2.FILLED)
    cv2.putText(sidebar, f'Free:  {free_pct}%', (194, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(sidebar, (180, 182), 5, (30, 30, 220), cv2.FILLED)
    cv2.putText(sidebar, f'Occ:   {occ_pct}%', (194, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)

    # Capacity horizontal dual-color progress bar (clean 21px padding below circle)
    bar_x, bar_y, bar_w, bar_h = 28, 226, width - 56, 8
    cv2.rectangle(sidebar, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 40, 50), cv2.FILLED)
    green_w = int(bar_w * (free_spaces / total_spaces)) if total_spaces > 0 else 0
    cv2.rectangle(sidebar, (bar_x, bar_y), (bar_x + green_w, bar_y + bar_h), (30, 210, 50), cv2.FILLED)
    cv2.rectangle(sidebar, (bar_x + green_w, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 220), cv2.FILLED)


    # 3. KPI Stat Cards (Available vs Occupied)
    card_y1, card_y2 = 254, 350
    card_w = (width - 40) // 2

    # Card 1: Available
    draw_card(sidebar, 15, card_y1, 15 + card_w, card_y2, bg_color=(24, 38, 28), border_color=(40, 120, 60))
    cv2.putText(sidebar, 'AVAILABLE', (25, card_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 230, 120), 1, cv2.LINE_AA)
    cv2.putText(sidebar, str(free_spaces), (25, card_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 240, 80), 3, cv2.LINE_AA)
    cv2.putText(sidebar, f'/ {total_spaces} BAYS', (25, card_y1 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 160), 1, cv2.LINE_AA)

    # Card 2: Occupied
    draw_card(sidebar, 25 + card_w, card_y1, width - 15, card_y2, bg_color=(38, 24, 28), border_color=(120, 40, 60))
    cv2.putText(sidebar, 'OCCUPIED', (35 + card_w, card_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(sidebar, str(occupied_spaces), (35 + card_w, card_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 240), 3, cv2.LINE_AA)
    cv2.putText(sidebar, f'{(occupied_spaces/total_spaces)*100:.1f}% OF LOT', (35 + card_w, card_y1 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 150, 150), 1, cv2.LINE_AA)

    # 4. Zone / Aisle Analysis Card
    draw_card(sidebar, 15, 362, width - 15, 514, bg_color=(30, 26, 34), border_color=(70, 65, 80))
    cv2.putText(sidebar, 'ZONE & AISLE OCCUPANCY', (28, 384), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    zones = [
        ('Zone A (West Bays)', [s for s in slot_details if s['x'] < 250]),
        ('Zone B (Central Bays)', [s for s in slot_details if 250 <= s['x'] < 650]),
        ('Zone C (East Bays)', [s for s in slot_details if s['x'] >= 650]),
    ]

    zy = 407
    for z_name, z_slots in zones:
        z_tot = len(z_slots)
        z_free = sum(1 for s in z_slots if s['is_free'])
        cv2.putText(sidebar, z_name, (28, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(sidebar, f'{z_free}/{z_tot} Free', (width - 92, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (50, 230, 90) if z_free > 0 else (100, 100, 230), 1, cv2.LINE_AA)

        # Mini progress bar
        mb_x, mb_y, mb_w, mb_h = 28, zy + 6, width - 56, 6
        cv2.rectangle(sidebar, (mb_x, mb_y), (mb_x + mb_w, mb_y + mb_h), (45, 40, 50), cv2.FILLED)
        z_fill = int(mb_w * (z_free / z_tot)) if z_tot > 0 else 0
        cv2.rectangle(sidebar, (mb_x, mb_y), (mb_x + z_fill, mb_y + mb_h), (30, 210, 50), cv2.FILLED)
        cv2.rectangle(sidebar, (mb_x + z_fill, mb_y), (mb_x + mb_w, mb_y + mb_h), (30, 30, 220), cv2.FILLED)
        zy += 34

    # 5. Real-Time Vacancy Trend Sparkline Card
    draw_card(sidebar, 15, 526, width - 15, 638, bg_color=(30, 26, 34), border_color=(70, 65, 80))
    cv2.putText(sidebar, 'REAL-TIME VACANCY TREND', (28, 546), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(sidebar, f'Now: {free_spaces} Free', (width - 105, 546), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 220, 80), 1, cv2.LINE_AA)

    trend_values = list(history_trend) if (history_trend and len(history_trend) > 1) else [free_spaces] * 20
    plot_x1, plot_y1, plot_w, plot_h = 28, 558, width - 56, 64
    cv2.rectangle(sidebar, (plot_x1, plot_y1), (plot_x1 + plot_w, plot_y1 + plot_h), (40, 36, 46), cv2.FILLED)
    cv2.line(sidebar, (plot_x1, plot_y1 + plot_h // 2), (plot_x1 + plot_w, plot_y1 + plot_h // 2), (55, 50, 60), 1, cv2.LINE_AA)

    min_v = max(0, min(trend_values) - 3)
    max_v = max(min_v + 6, max(trend_values) + 3)
    val_range = max(1, max_v - min_v)

    pts = []
    for i, val in enumerate(trend_values):
        px = int(plot_x1 + (i / (len(trend_values) - 1)) * plot_w)
        norm = (val - min_v) / val_range
        py = int(plot_y1 + plot_h - (norm * (plot_h - 10)) - 5)
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(sidebar, pts[i], pts[i + 1], (50, 220, 80), 2, cv2.LINE_AA)
    cv2.circle(sidebar, pts[-1], 4, (0, 255, 120), cv2.FILLED)

    # 6. Footer Controls Card
    draw_card(sidebar, 15, 650, width - 15, 706, bg_color=(28, 24, 32), border_color=(60, 55, 70))
    cv2.putText(sidebar, '[Q] Quit  |  [P] Pause  |  [T] Labels  |  [S] Sidebar', (22, 672),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(sidebar, 'THRESHOLD: 900 px  |  CV-ALGO: ADAPTIVE', (22, 693),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    return sidebar


def run_parking_counter(video_path=None, pos_path=None):
    """
    Run the live parking counter video feed loop with interactive OpenCV display.
    Features:
    - Left Sidebar with live Donut gauge, KPI cards, Zone breakdown, and Trend sparkline.
    - Keyboard controls:
        [Q] / [ESC]: Quit
        [P] / [SPACE]: Pause / Resume playback
        [T]: Toggle numeric pixel count labels inside slots
        [S]: Toggle dashboard sidebar on/off
    
    Args:
        video_path (str or Path, optional): Path to input parking lot video.
        pos_path (str or Path, optional): Path to CarParkPos pickle file.
    """
    from collections import deque

    default_vid, default_pos = get_default_paths()
    video_file = str(video_path if video_path is not None else default_vid)
    pos_file = str(pos_path if pos_path is not None else default_pos)

    # Load parking positions
    pos_list = load_parking_positions(pos_file)
    if not pos_list:
        print("[ERROR] No parking positions found. Please run ParkingSpacePicker.py first!")
        return

    # Open video feed
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_file}")
        return

    window_name = "Smart Parking Space Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1480, 720)

    print("=" * 60)
    print(" Smart Parking Space Counter - Live Demo")
    print(f" - Video Source: {video_file}")
    print(f" - Slots Loaded: {len(pos_list)}")
    print(" - Controls:")
    print("     [Q] / [ESC] : Quit")
    print("     [P] / [SPACE]: Pause / Resume")
    print("     [T]         : Toggle Slot Numbers")
    print("     [S]         : Toggle Sidebar")
    print("=" * 60)

    # Rolling history of free spaces for real-time sparkline graph
    history_trend = deque(maxlen=50)
    show_labels = True
    show_sidebar = True
    is_paused = False
    last_frame = None

    while True:
        if not is_paused:
            # Loop video continuously
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            success, frame = cap.read()
            if not success or frame is None:
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else None
            if frame is None:
                break

        # Process frame to detect edges
        dilated = preprocess_image(frame)

        # Classify each spot and draw rectangles
        free_spaces, total_spaces, slot_details = check_parking_spaces(
            dilated, frame, pos_list, show_labels=show_labels
        )

        if not is_paused:
            history_trend.append(free_spaces)

        # Assemble display canvas with Left Sidebar Dashboard
        if show_sidebar:
            sidebar = render_dashboard_sidebar(
                height=frame.shape[0],
                free_spaces=free_spaces,
                total_spaces=total_spaces,
                slot_details=slot_details,
                history_trend=history_trend,
                width=380
            )
            display_canvas = np.hstack([sidebar, frame])
        else:
            display_canvas = frame

        # Show frame
        cv2.imshow(window_name, display_canvas)

        # Handle keyboard input
        wait_time = 30 if is_paused else 10
        key = cv2.waitKey(wait_time) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[INFO] Exiting video feed...")
            break
        elif key in (ord('p'), ord('P'), ord(' ')):
            is_paused = not is_paused
            print(f"[INFO] Playback {'PAUSED' if is_paused else 'RESUMED'}")
        elif key in (ord('t'), ord('T')):
            show_labels = not show_labels
            print(f"[INFO] Slot Labels: {'ENABLED' if show_labels else 'DISABLED'}")
        elif key in (ord('s'), ord('S')):
            show_sidebar = not show_sidebar
            print(f"[INFO] Sidebar Dashboard: {'ENABLED' if show_sidebar else 'DISABLED'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_parking_counter()


