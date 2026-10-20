
import time
from camera import MantaCamera

NUM_FRAMES = 100

with MantaCamera() as camera:
    camera.configure(exposure_us=5000, gain=None)

    print("[TEST] Camera-only acquisition")
    print("[TEST] No processing or display")

    # Warm up
    for _ in range(5):
        camera.read()

    start = time.perf_counter()

    for _ in range(NUM_FRAMES):
        camera.read()

    elapsed = time.perf_counter() - start

    fps = NUM_FRAMES / elapsed

    print(f"Frames: {NUM_FRAMES}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Acquisition FPS: {fps:.2f}")
