from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

detector = Detector(classes = [1,2,3,5,7]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./yolov7x.pt',) # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video("./IO_data/input/video/city_1.mp4", output="./IO_data/output/city_1.mp4", show_live = True, skip_frames = 400, count_objects = True, verbose=0, data_gathering=True)