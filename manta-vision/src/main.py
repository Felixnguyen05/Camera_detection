
import cv2
import time
import queue
from pathlib import Path

from vmbpy import FrameStatus, PixelFormat

from camera import MantaCamera
from processing import process_frame


CAMERA_ID = None
EXPOSURE_US = 5000
DISPLAY_SCALE = 0.75
ENABLE_PROCESSING = True

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # Keep only the newest frame.
    frame_queue = queue.Queue(maxsize=1)

    received_count = 0

    def frame_handler(cam, stream, frame):
        nonlocal received_count

        try:
            if frame.get_status() != FrameStatus.Complete:
                return

            if frame.get_pixel_format() != PixelFormat.Mono8:
                return

            # Copy image BEFORE requeuing SDK buffer.
            image = frame.as_opencv_image().copy()
            received_count += 1

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

        except Exception as exc:
            print("[CALLBACK ERROR]", exc)

        finally:
            # Always return the original frame to VmbPy.
            cam.queue_frame(frame)

    cv2.namedWindow("Manta Live", cv2.WINDOW_NORMAL)

    with MantaCamera(camera_id=CAMERA_ID) as camera:
        # camera.optimize_network()

        camera.configure(
            exposure_us=EXPOSURE_US,
            gain=None
        )

        print("[INFO] Starting async streaming")

        camera.cam.start_streaming(
            handler=frame_handler,
            buffer_count=5
        )

        last_frame = None
        last_fps_time = time.perf_counter()
        last_received = 0
        camera_fps = 0.0

        try:
            while True:
                try:
                    image = frame_queue.get(timeout=0.5)
                    last_frame = image

                except queue.Empty:
                    print("[WARNING] No frame received")
                    continue

                now = time.perf_counter()

                if now - last_fps_time >= 1:
                    camera_fps = (
                        received_count - last_received
                    ) / (now - last_fps_time)

                    last_received = received_count
                    last_fps_time = now

                # Display the original mono image.
                display = last_frame.copy()

                cv2.putText(
                    display,
                    f"Camera FPS: {camera_fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    255,
                    2
                )

                if DISPLAY_SCALE != 1.0:
                    display = cv2.resize(
                        display,
                        None,
                        fx=DISPLAY_SCALE,
                        fy=DISPLAY_SCALE,
                        interpolation=cv2.INTER_AREA
                    )

                cv2.imshow("Manta Live", display)

                if ENABLE_PROCESSING:
                    bgr = cv2.cvtColor(
                        last_frame,
                        cv2.COLOR_GRAY2BGR
                    )

                    edges = process_frame(bgr)

                    cv2.imshow(
                        "Edge Detection",
                        edges
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("s"):
                    filename = (
                        OUTPUT_DIR /
                        f"capture_{time.time_ns()}.png"
                    )

                    cv2.imwrite(
                        str(filename),
                        last_frame
                    )

                    print("[SAVED]", filename)

        finally:
            print("[INFO] Stopping streaming")
            camera.cam.stop_streaming()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
