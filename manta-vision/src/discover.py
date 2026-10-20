
from vmbpy import VmbSystem

def main():
    with VmbSystem.get_instance() as vmb:
        cameras = vmb.get_all_cameras()

        if not cameras:
            print("No cameras detected.")
            return

        for index, cam in enumerate(cameras):
            print(f"Camera {index}")
            print(f"  ID: {cam.get_id()}")
            print(f"  Model: {cam.get_model()}")
            print(f"  Serial: {cam.get_serial()}")
            print(f"  Name: {cam.get_name()}")

if __name__ == "__main__":
    main()
