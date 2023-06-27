from src.detection_helpers import *
from src.tracking_helpers import *
from src.bridge_wrapper import *
from PIL import Image
import os

detector = Detector(
    classes=None
)  # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model(
    "./epoch_074.pt",
)  # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(
    reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector
)
i = 0
# output = None will not save the output video
for file in os.listdir("./IO_data/input/video"):
    file = "test_video.mp4"
    i += 1
    tracker.track_video(
        f"./IO_data/input/video/{file}",
        output=f"./IO_data/output/{file}",
        show_live=True,
        skip_frames=2,
        count_objects=True,
        verbose=2,
        data_gathering=False,
        data_file=f"IO_data/output/data/vehicle_data_{i}.csv",
    )
    break
