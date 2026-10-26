
import cv2
import time
import queue
import numpy as np
from pathlib import Path

from vmbpy import FrameStatus, PixelFormat
from camera import MantaCamera


# ============================================
# CONFIGURATION
# ============================================

CAMERA_ID = None
EXPOSURE_US = 5000

MIN_AREA = 5000
DISPLAY_SCALE = 0.8

# Start with bright objects on darker background
THRESHOLD = 160
KERNEL_SIZE = 41

TEMPLATE_PATH = Path("templates/bracket.png")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================
# FULL COMPONENT SEGMENTATION
# ============================================

def detect_full_component(
    gray,
    threshold_value,
    kernel_size
):
    # 1. Reduce camera noise
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # 2. Detect bright component regions
    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # 3. Connect bright regions belonging
    #    to the same component
    size = max(3, kernel_size | 1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size)
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 4. Detect outer contours
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, closed

    # 5. Select the largest valid region
    height, width = gray.shape

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore regions touching the image border
        if (
            x <= 1 or y <= 1 or
            x + w >= width - 1 or
            y + h >= height - 1
        ):
            continue

        candidates.append(contour)

    if not candidates:
        return None, closed

    contour = max(
        candidates,
        key=cv2.contourArea
    )

    # 6. Fill internal holes
    component_mask = np.zeros_like(gray)

    cv2.drawContours(
        component_mask,
        [contour],
        -1,
        255,
        cv2.FILLED
    )

    # 7. Re-extract full external contour
    full_contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not full_contours:
        return None, component_mask

    contour = max(
        full_contours,
        key=cv2.contourArea
    )

    # 8. Geometric center
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None, component_mask

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # 9. PCA-like orientation from area moments
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    theta = 0.5 * np.arctan2(
        2 * mu11,
        mu20 - mu02
    )

    angle_deg = np.degrees(theta) % 180

    # 10. Rotated bounding rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    result = {
        "contour": contour,
        "center": (float(cx), float(cy)),
        "angle": float(angle_deg),
        "box": box,
        "area": cv2.contourArea(contour),
        "mask": component_mask
    }

    return result, component_mask


# ============================================
# OPTIONAL SHAPE MATCHING
# ============================================

def load_template_contour(path, threshold, kernel_size):
    if not path.exists():
        print("[INFO] No template found.")
        print("[INFO] Running full-shape detection only.")
        return None

    template = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if template is None:
        return None

    result, _ = detect_full_component(
        template,
        threshold,
        kernel_size
    )

    if result is None:
        print("[WARNING] Template segmentation failed.")
        return None

    print("[INFO] Shape template loaded.")
    return result["contour"]


# ============================================
# DRAW DETECTION
# ============================================

def draw_result(frame, result, processing_ms, match_distance):
    display = cv2.cvtColor(
        frame,
        cv2.COLOR_GRAY2BGR
    )

    if result is None:
        cv2.putText(
            display,
            "NO COMPONENT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return display

    contour = result["contour"]
    cx, cy = result["center"]
    angle = result["angle"]

    center = (int(cx), int(cy))

    # Full outer shape in green
    cv2.drawContours(
        display,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # Rotated bounding box in cyan
    cv2.drawContours(
        display,
        [result["box"]],
        -1,
        (255, 255, 0),
        2
    )

    # Object center in red
    cv2.circle(
        display,
        center,
        6,
        (0, 0, 255),
        -1
    )

    # Orientation axis in magenta
    rad = np.radians(angle)
    length = 100

    dx = int(length * np.cos(rad))
    dy = int(length * np.sin(rad))

    cv2.line(
        display,
        (center[0] - dx, center[1] - dy),
        (center[0] + dx, center[1] + dy),
        (255, 0, 255),
        3
    )

    texts = [
        "FULL COMPONENT DETECTED",
        f"Position: ({cx:.1f}, {cy:.1f}) px",
        f"Orientation axis: {angle:.1f} deg",
        f"Area: {result['area']:.0f} px2",
        f"Processing: {processing_ms:.1f} ms"
    ]

    if match_distance is not None:
        texts.append(
            f"Shape distance: {match_distance:.4f}"
        )

    for i, text in enumerate(texts):
        cv2.putText(
            display,
            text,
            (20, 35 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 210, 0),
            2
        )

    return display


# ============================================
# MAIN CAMERA APPLICATION
# ============================================

def main():
    frame_queue = queue.Queue(maxsize=1)

    threshold_value = THRESHOLD
    kernel_size = KERNEL_SIZE

    show_mask = True
    template_contour = load_template_contour(
        TEMPLATE_PATH,
        threshold_value,
        kernel_size
    )

    def frame_handler(cam, stream, frame):
        try:
            if frame.get_status() != FrameStatus.Complete:
                return

            if frame.get_pixel_format() != PixelFormat.Mono8:
                return

            image = frame.as_opencv_image().copy()

            try:
                frame_queue.put_nowait(image)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    frame_queue.put_nowait(image)
                except queue.Full:
                    pass

        finally:
            cam.queue_frame(frame)

    print("CONTROLS")
    print("Q / ESC : Quit")
    print("S : Save current image and mask")
    print("M : Toggle binary mask window")
    print("+ : Increase threshold")
    print("- : Decrease threshold")
    print("] : Increase closing kernel")
    print("[ : Decrease closing kernel")

    with MantaCamera(camera_id=CAMERA_ID) as camera:
        camera.configure(
            exposure_us=EXPOSURE_US,
            gain=None
        )

        camera.cam.start_streaming(
            handler=frame_handler,
            buffer_count=5
        )

        try:
            while True:
                try:
                    frame = frame_queue.get(
                        timeout=0.5
                    )
                except queue.Empty:
                    continue

                start = time.perf_counter()

                result, mask = detect_full_component(
                    frame,
                    threshold_value,
                    kernel_size
                )

                match_distance = None

                if (
                    result is not None and
                    template_contour is not None
                ):
                    match_distance = cv2.matchShapes(
                        template_contour,
                        result["contour"],
                        cv2.CONTOURS_MATCH_I1,
                        0.0
                    )

                processing_ms = (
                    time.perf_counter() - start
                ) * 1000

                display = draw_result(
                    frame,
                    result,
                    processing_ms,
                    match_distance
                )

                cv2.putText(
                    display,
                    f"Threshold: {threshold_value} "
                    f"Kernel: {kernel_size}",
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                if DISPLAY_SCALE != 1.0:
                    display = cv2.resize(
                        display,
                        None,
                        fx=DISPLAY_SCALE,
                        fy=DISPLAY_SCALE
                    )

                cv2.imshow(
                    "Manta - Full Component Detection",
                    display
                )

                if show_mask:
                    mask_display = cv2.resize(
                        mask,
                        None,
                        fx=DISPLAY_SCALE,
                        fy=DISPLAY_SCALE
                    )

                    cv2.imshow(
                        "Full Component Mask",
                        mask_display
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("s"):
                    timestamp = time.time_ns()

                    cv2.imwrite(
                        str(OUTPUT_DIR / f"component_{timestamp}.png"),
                        frame
                    )

                    cv2.imwrite(
                        str(OUTPUT_DIR / f"mask_{timestamp}.png"),
                        mask
                    )

                    print("[SAVED] Image and mask")

                elif key == ord("m"):
                    show_mask = not show_mask

                    if not show_mask:
                        cv2.destroyWindow(
                            "Full Component Mask"
                        )

                elif key in (ord("+"), ord("=")):
                    threshold_value = min(
                        255, threshold_value + 5
                    )
                    print("Threshold:", threshold_value)

                elif key == ord("-"):
                    threshold_value = max(
                        0, threshold_value - 5
                    )
                    print("Threshold:", threshold_value)

                elif key == ord("]"):
                    kernel_size = min(
                        101, kernel_size + 10
                    )
                    print("Kernel:", kernel_size)

                elif key == ord("["):
                    kernel_size = max(
                        3, kernel_size - 10
                    )
                    print("Kernel:", kernel_size)

        finally:
            camera.cam.stop_streaming()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
