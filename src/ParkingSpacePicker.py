"""
Parking Space Picker Tool
=========================
An interactive tool to mark and calibrate parking space bounding boxes.

Instructions:
- Left-Click on parking lot image: Mark a new parking space (top-left corner).
- Right-Click inside an existing box: Remove that parking space.
- Press [Q] or [ESC]: Save calibrated spots to file and exit.
"""

from pathlib import Path
import pickle
import cv2

# Parking spot dimensions (width, height in pixels)
SLOT_WIDTH = 107
SLOT_HEIGHT = 48


def get_default_paths():
    """
    Resolve default paths for the parking lot image and positions file.
    
    Returns:
        tuple[Path, Path]: (image_path, positions_path)
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    candidate_img_paths = [
        project_root / "asset" / "carParkImg.png",
        current_dir / "asset" / "carParkImg.png",
        Path("asset/carParkImg.png")
    ]
    candidate_pos_paths = [
        project_root / "asset" / "CarParkPos",
        current_dir / "asset" / "CarParkPos",
        Path("asset/CarParkPos")
    ]

    img_path = next((p for p in candidate_img_paths if p.exists()), candidate_img_paths[0])
    pos_path = next((p for p in candidate_pos_paths if p.exists()), candidate_pos_paths[0])

    return img_path, pos_path


def load_positions(pos_path):
    """
    Load saved parking slot coordinates from disk.
    
    Args:
        pos_path (Path or str): Path to the CarParkPos pickle file.
        
    Returns:
        list of tuple[int, int]: Loaded (x, y) coordinates.
    """
    pos_file = Path(pos_path)
    if pos_file.exists():
        try:
            with open(pos_file, "rb") as file:
                positions = pickle.load(file)
                print(f"[INFO] Loaded {len(positions)} existing spots from {pos_file.name}")
                return positions
        except Exception as error:
            print(f"[WARNING] Could not read existing positions: {error}")
    return []


def save_positions(pos_path, pos_list):
    """
    Save parking slot coordinates to disk using pickle.
    
    Args:
        pos_path (Path or str): Destination file path.
        pos_list (list): List of (x, y) coordinates.
    """
    pos_file = Path(pos_path)
    pos_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(pos_file, "wb") as file:
            pickle.dump(pos_list, file)
        print(f"[INFO] Successfully saved {len(pos_list)} spots to {pos_file}")
    except Exception as error:
        print(f"[ERROR] Failed to save positions: {error}")


def draw_hud(image, total_spots):
    """
    Draw a clean, beginner-friendly instruction banner on the display canvas.
    
    Args:
        image (np.ndarray): Target image.
        total_spots (int): Current count of marked spaces.
    """
    # Background HUD card
    cv2.rectangle(image, (15, 15), (540, 75), (35, 35, 35), cv2.FILLED)
    cv2.rectangle(image, (15, 15), (540, 75), (255, 180, 0), 2)

    # Title & Count
    title = f"Parking Space Picker | Total Spots: {total_spots}"
    cv2.putText(
        image,
        title,
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Instructions
    instructions = "Left-Click: Add | Right-Click: Remove | [Q] / [ESC]: Save & Exit"
    cv2.putText(
        image,
        instructions,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


def run_picker(img_path=None, pos_path=None):
    """
    Launch interactive OpenCV window to annotate and modify parking spots.
    
    Args:
        img_path (str or Path, optional): Path to reference parking lot image.
        pos_path (str or Path, optional): Path to CarParkPos pickle file.
    """
    default_img, default_pos = get_default_paths()
    img_file = Path(img_path if img_path is not None else default_img)
    pos_file = Path(pos_path if pos_path is not None else default_pos)

    if not img_file.exists():
        print(f"[ERROR] Parking image not found: {img_file}")
        return

    # Load existing positions
    pos_list = load_positions(pos_file)

    # Mouse callback closure
    def on_mouse_click(event, x, y, flags, params):
        # Left click: Add a new parking spot at (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            pos_list.append((x, y))
            save_positions(pos_file, pos_list)

        # Right click: Check if clicked inside an existing spot and remove it
        elif event == cv2.EVENT_RBUTTONDOWN:
            for index, (pos_x, pos_y) in enumerate(pos_list):
                if pos_x < x < pos_x + SLOT_WIDTH and pos_y < y < pos_y + SLOT_HEIGHT:
                    pos_list.pop(index)
                    save_positions(pos_file, pos_list)
                    break

    window_name = "Parking Space Picker - Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("=" * 60)
    print(" Parking Space Picker")
    print(f" - Image: {img_file.name}")
    print(f" - Saving to: {pos_file.name}")
    print(" - Left-Click: Add slot")
    print(" - Right-Click: Remove slot")
    print(" - Press [Q] or [ESC] to Exit")
    print("=" * 60)

    while True:
        # Load fresh image each loop to redraw bounding boxes
        canvas = cv2.imread(str(img_file))
        if canvas is None:
            print(f"[ERROR] Failed to read image from {img_file}")
            break

        # Draw all currently registered parking slots
        for index, (x, y) in enumerate(pos_list):
            cv2.rectangle(
                canvas,
                (x, y),
                (x + SLOT_WIDTH, y + SLOT_HEIGHT),
                (255, 0, 255),
                2
            )
            # Spot number tag
            cv2.putText(
                canvas,
                str(index + 1),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Draw on-screen HUD instruction banner
        draw_hud(canvas, len(pos_list))

        cv2.imshow(window_name, canvas)

        # Handle keyboard input
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print(f"[INFO] Finished! Total {len(pos_list)} spots saved.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_picker()

