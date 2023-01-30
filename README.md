# Vehicle-Trajectory-Prediction-Veh2Veh
This is the final project for DS201 - Deep Learning for Data Science of the University of Information Technology course.

The purpose of this project is to predict the trajectory of the vehicles in Thu Duc District, Ho Chi Minh city, Vietnam. There are 3 main sub-tasks for this project. 
There are Vehicle Detection, Vehicle Tracking and Vehicle Trajectory Prediction.

I choose YOLOv7 for Vehicle Detection Task, DeepSORT for Vehicle Tracking Task and CNN-LSTM/CNN-GRU for Vehicle Trajectory Task.

### Tech Stack:
Pytorch

Tensorflow

Python

OpenCV

Scipy

Pillow

### Demo


Uploading city_new_3.mp4

You can see in the small demo. The center of every bounding boxes is the current state, the model is trying to predict the center point in the next state.

In this project, we use Time Series Approach to predict the trajectory of the vehicle
