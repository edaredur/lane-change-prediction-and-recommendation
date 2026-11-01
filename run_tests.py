import subprocess

def run_commands():
    command1 = 'sudo python3 Program_Main_test.py --source "/home/lanechange/Videos/raw_webcam_006_5_FPS.mp4" --yolo-skip 1 --resize 360 --lane-skip 1'
    command2 = 'sudo python3 Program_Main_test.py --source "/home/lanechange/Videos/raw_webcam_006_10_FPS.mp4" --yolo-skip 1 --resize 360 --lane-skip 1'
    command3 = 'sudo python3 Program_Main_test.py --source "/home/lanechange/Videos/raw_webcam_006_15_FPS.mp4" --yolo-skip 1 --resize 360 --lane-skip 1'
    command4 = 'sudo python3 Program_Main_test.py --source "/home/lanechange/Videos/raw_webcam_006_30_FPS.mp4" --yolo-skip 1 --resize 360 --lane-skip 1'

    print(f"Running first command:\n{command1}")
    subprocess.run(command1, shell=True, check=True)
    print("First command completed successfully.\n")

    print(f"Running second command:\n{command2}")
    subprocess.run(command2, shell=True, check=True)
    print("Second command completed successfully.\n")

    print(f"Running third command:\n{command3}")
    subprocess.run(command3, shell=True, check=True)
    print("Third command completed successfully.\n")

    print(f"Running fourth command:\n{command4}")
    subprocess.run(command4, shell=True, check=True)
    print("Fourth command completed successfully.\n")

    print("All commands completed successfully.")

if __name__ == "__main__":
    run_commands()
