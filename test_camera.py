
from vmbpy import VmbSystem

with VmbSystem.get_instance() as vmb:
    cameras = vmb.get_all_cameras()

    print(f"Detected cameras: {len(cameras)}")

    for camera in cameras:
        print("Camera ID:", camera.get_id())
        print("Model:", camera.get_model())
