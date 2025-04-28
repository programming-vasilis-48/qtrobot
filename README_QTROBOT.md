# QTrobot Confusion Detection System - Installation Guide

This guide provides step-by-step instructions for installing and running the confusion detection system on QTrobot.

## Installation

1. Pull the latest changes from the repository:

```bash
cd ~/catkin_ws/src/vasilis/qtrobot
git pull
```

2. Replace the face detector and feature extractor nodes with the real face detection versions:

```bash
cd ~/catkin_ws/src/vasilis/qtrobot/src/qt_confusion_detection/src
mv face_detector_node_real.py face_detector_node.py
mv feature_extractor_node_fixed.py feature_extractor_node.py
chmod +x face_detector_node.py
chmod +x feature_extractor_node.py
chmod +x confusion_classifier_node.py
cd ~/catkin_ws/src/vasilis/qtrobot/src/qt_repair_policy/src
chmod +x policy_engine_node.py
```

Make sure the Haar cascade file exists in the data directory:
```bash
cd ~/catkin_ws/src/vasilis/qtrobot/src/qt_confusion_detection/src
ls -l data/haarcascade_frontalface_default.xml
```

If the file doesn't exist, download it:
```bash
mkdir -p data
wget -O data/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

3. Build the packages:

```bash
cd ~/catkin_ws
catkin_make
```

4. Source the workspace:

```bash
source ~/catkin_ws/devel/setup.bash
```

## Running the System

To start the confusion detection system, use the full path to the launch file:

```bash
roslaunch ~/catkin_ws/src/vasilis/qtrobot/launch/confusion_detection.launch
```

By default, confusion simulation is disabled. If you want to enable it (for testing purposes), use:

```bash
roslaunch ~/catkin_ws/src/vasilis/qtrobot/launch/confusion_detection.launch simulate_confusion:=true
```

## Troubleshooting

### Common Errors

#### Launch File Not Found

If you get an error like:
```
RLException: [confusion_detection.launch] is neither a launch file in package [qtrobot] nor is [qtrobot] a launch file name
```

Always use the full path to the launch file:
```bash
roslaunch ~/catkin_ws/src/vasilis/qtrobot/launch/confusion_detection.launch
```

#### Face Detector Errors

If you see errors related to the face detector, make sure you've replaced the original face detector with the fixed version as described in the installation steps.

#### Feature Extractor Errors

If you see errors like:
```
Error processing features: 'tuple' object does not support item assignment
```

Make sure you've replaced the feature extractor with the fixed version as described in the installation steps.

## System Overview

The system consists of the following components:

1. **Face Detector Node**: Detects faces in the camera feed and extracts basic facial features
2. **Feature Extractor Node**: Processes facial features to extract higher-level features for confusion detection
3. **Confusion Classifier Node**: Classifies whether a human is confused based on processed facial features
4. **Policy Engine Node**: Selects and executes appropriate repair strategies in response to detected confusion

## Topics

The system publishes and subscribes to the following topics:

- **Input**:
  - `/camera/color/image_raw` (sensor_msgs/Image): Camera feed from QTrobot

- **Output**:
  - `/robot/speech/say` (std_msgs/String): Text for the robot to speak
  - `/robot/behavior/trigger` (std_msgs/String): Behavior triggers for the robot
