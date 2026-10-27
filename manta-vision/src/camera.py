
import cv2

from vmbpy import (
    VmbSystem,
    PixelFormat,
    intersect_pixel_formats,
)


class MantaCamera:
    def __init__(self, camera_id=None):
        self.camera_id = camera_id
        self.vmb = None
        self.cam = None

    # Connect camera
    def __enter__(self):
        self.vmb = VmbSystem.get_instance()
        self.vmb.__enter__()

        try:
            cameras = self.vmb.get_all_cameras()

            if not cameras:
                raise RuntimeError(
                    "No Allied Vision camera detected."
                )

            if self.camera_id:
                self.cam = self.vmb.get_camera_by_id(
                    self.camera_id
                )
            else:
                self.cam = cameras[0]

            self.cam.__enter__()

            print("[CAMERA] Connected")
            print("[CAMERA] Model:", self.cam.get_model())
            print("[CAMERA] ID:", self.cam.get_id())

            supported = self.cam.get_pixel_formats()

            print(
                "[CAMERA] Supported formats:",
                supported
            )

            # Manta G-125B supports Mono8
            preferred = (
                PixelFormat.Mono8,
                PixelFormat.Bgr8,
                PixelFormat.BayerRG8,
                PixelFormat.BayerGR8,
                PixelFormat.BayerGB8,
                PixelFormat.BayerBG8,
            )

            available = intersect_pixel_formats(
                supported,
                preferred
            )

            if not available:
                raise RuntimeError(
                    "No compatible pixel format found."
                )

            selected = next(
                fmt for fmt in preferred
                if fmt in available
            )

            self.cam.set_pixel_format(selected)

            print(
                "[CAMERA] Selected pixel format:",
                selected
            )

            return self

        except Exception:
            self.__exit__(None, None, None)
            raise

    # CONFIGURE CAMERA
    def configure(self, exposure_us=None, gain=None):

        # Disable hardware trigger
        try:
            self.cam.TriggerMode.set("Off")
            print("[CAMERA] Trigger disabled")
        except Exception as exc:
            print("[WARNING] Trigger:", exc)

        # Configure exposure
        if exposure_us is not None:

            try:
                self.cam.ExposureAuto.set("Off")
            except Exception:
                pass

            exposure_feature = None

            for name in (
                "ExposureTimeAbs",
                "ExposureTime",
            ):
                feature = getattr(
                    self.cam,
                    name,
                    None
                )

                if feature is not None:
                    exposure_feature = feature
                    print(
                        "[CAMERA] Exposure feature:",
                        name
                    )
                    break

            if exposure_feature is not None:
                try:
                    exposure_feature.set(
                        float(exposure_us)
                    )

                    print(
                        f"[CAMERA] Exposure set: "
                        f"{exposure_us} us"
                    )

                except Exception as exc:
                    print(
                        "[WARNING] Exposure setting:",
                        exc
                    )
            else:
                print(
                    "[WARNING] No exposure feature found."
                )

        # Configure gain
        if gain is not None:

            try:
                self.cam.GainAuto.set("Off")
            except Exception:
                pass

            for name in ("Gain", "GainRaw"):
                feature = getattr(
                    self.cam,
                    name,
                    None
                )

                if feature is not None:
                    try:
                        feature.set(gain)

                        print(
                            f"[CAMERA] {name} set: {gain}"
                        )
                    except Exception as exc:
                        print(
                            "[WARNING] Gain setting:",
                            exc
                        )
                    break

    # Read frame
    def read(self):

        frame = self.cam.get_frame(
            timeout_ms=3000
        )

        pixel_format = frame.get_pixel_format()

        # Camera already outputs BGR8
        if pixel_format == PixelFormat.Bgr8:
            image = frame.as_opencv_image().copy()

        # Manta G-125B monochrome format
        elif pixel_format == PixelFormat.Mono8:

            gray = frame.as_opencv_image()

            image = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

        # Bayer or other convertible formats
        else:
            converted = frame.convert_pixel_format(
                PixelFormat.Bgr8
            )

            image = converted.as_opencv_image().copy()

        return image

    # RELEASE CAMERA
    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        try:
            if self.cam is not None:
                self.cam.__exit__(
                    exc_type,
                    exc_value,
                    traceback
                )
        finally:
            self.cam = None

            if self.vmb is not None:
                try:
                    self.vmb.__exit__(
                        exc_type,
                        exc_value,
                        traceback
                    )
                finally:
                    self.vmb = None