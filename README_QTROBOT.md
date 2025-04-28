# QTrobot Confusion Detection System - Installation Guide

This guide provides step-by-step instructions for installing and running the confusion detection system on QTrobot.

## Installation

1. Clone the repository into your catkin workspace:

```bash
cd ~/catkin_ws/src
git clone https://github.com/yourusername/qt_confusion_detection.git
```

2. Replace the face detector node with the fixed version:

```bash
cd ~/catkin_ws/src/qt_confusion_detection/src/qt_confusion_detection/src
mv face_detector_node_fixed.py face_detector_node.py
chmod +x face_detector_node.py
chmod +x feature_extractor_node.py
chmod +x confusion_classifier_node.py
cd ~/catkin_ws/src/qt_confusion_detection/src/qt_repair_policy/src
chmod +x policy_engine_node.py
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

To start the confusion detection system:

```bash
roslaunch qt_confusion_detection confusion_detection.launch
```

If you encounter an error finding the launch file, you can use the full path:

```bash
roslaunch ~/catkin_ws/src/qt_confusion_detection/launch/confusion_detection.launch
```

## Troubleshooting

### Launch File Not Found

If you get an error like:
```
RLException: [confusion_detection.launch] is neither a launch file in package [qt_confusion_detection] nor is [qt_confusion_detection] a launch file name
```

Try these solutions:

1. Make sure you've sourced your workspace:
```bash
source ~/catkin_ws/devel/setup.bash
```

2. Use the full path to the launch file:
```bash
roslaunch ~/catkin_ws/src/qt_confusion_detection/launch/confusion_detection.launch
```

3. Check if the package is properly installed:
```bash
rospack find qt_confusion_detection
```

### Face Detector Errors

If you see errors related to the face detector, make sure you've replaced the original face detector with the fixed version as described in the installation steps.

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
