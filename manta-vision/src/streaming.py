"""Camera acquisition only: Mono8 frames, newest-frame queue, VmbPy buffer requeue."""
from queue import Empty, Full, Queue
from vmbpy import FrameStatus, PixelFormat
import numpy as np


class LatestFrameStream:
    def __init__(self):
        """Create a one-frame queue and initialize acquisition counters.

        Using only the newest frame avoids growing latency if image analysis
        is slower than the camera acquisition rate."""
        self.queue = Queue(maxsize=1)
        self.received = 0
        self.skipped = 0
        self.errors = 0

    def callback(self, cam, stream, frame):
        """VmbPy callback invoked for every delivered camera frame.

        The callback copies a completed Mono8 frame *before* giving the SDK
        its buffer back. It replaces any older queued image with the newest
        frame and ALWAYS requeues the SDK frame in finally.

        This callback runs in the SDK acquisition thread; keep it fast and
        do not perform ArUco detection or disk writes inside it."""
        try:
            if frame.get_status() != FrameStatus.Complete or frame.get_pixel_format() != PixelFormat.Mono8:
                self.skipped += 1
                return
            # SDK buffers are reused; copy before requeue_frame is called.
            image = frame.as_opencv_image().copy()
            # VmbPy can represent Mono8 as (H, W, 1) instead of (H, W).
            # Both are monochrome, not RGB. Remove only the singleton axis.
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
            if image.ndim != 2 or image.dtype != np.uint8:
                raise ValueError(f'Unexpected Mono8 array: shape={image.shape}, dtype={image.dtype}')
            if self.received == 0:
                print(f'[STREAM] Mono8 array shape={image.shape}, dtype={image.dtype}')
            self.received += 1
            try:
                self.queue.put_nowait(image)
            except Full:
                # Drop the OLD image so processing stays close to real time.
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(image)
                except Full:
                    pass
        except Exception as exc:
            self.errors += 1
            if self.errors <= 5:
                print('[STREAM ERROR]', repr(exc))
        finally:
            # Critical: without this the camera eventually runs out of buffers.
            cam.queue_frame(frame)

    def get(self, timeout=0.5):
        """Wait at most ``timeout`` seconds for the latest copied frame.

        Returns a 2-D uint8 image or None if no frame arrived. A timeout
        does not necessarily mean the camera is disconnected."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
