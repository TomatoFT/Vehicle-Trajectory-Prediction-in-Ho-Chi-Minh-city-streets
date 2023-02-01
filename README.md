# Vehicle Trajectory Prediction in Ho Chi Minh City streets.

This is the final project for DS201 - Deep Learning for Data Science of the University of Information Technology course.

The purpose of this project is to predict the trajectory of the vehicles in Thu Duc District, Ho Chi Minh city, Vietnam. There are 3 main sub-tasks for this project. 
There are Vehicle Detection, Vehicle Tracking and Vehicle Trajectory Prediction.

I choose YOLOv7 for Vehicle Detection Task, DeepSORT for Vehicle Tracking Task and CNN-LSTM/CNN-GRU for Vehicle Trajectory Task.

### Tech Stack:
<p>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="50px" width="50px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" height="50px" width="50px" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="50px" width="50px" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg"  height="50px" width="50px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" height="50px" width="50px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/anaconda/anaconda-original.svg" height="50px" width="50px"/>

</p>


### Demo



https://user-images.githubusercontent.com/79329526/215509399-ed339285-a138-4243-a7f9-5b04b788e0aa.mp4



You can see in the small demo. The center of every bounding boxes is the current state, the model is trying to predict the center point in the next state.

In this project, we use Time Series Approach to predict the trajectory of the vehicle.


### Set up
Download the pretrained weight of YoloV7 in this link: https://github.com/WongKinYiu/yolov7 

Go to `IO_data/input/video` folder to store the download or recorded of the vehicle on the street. 

Go to `run.py` to change some hyperparamater of function __tracker.track_video__
