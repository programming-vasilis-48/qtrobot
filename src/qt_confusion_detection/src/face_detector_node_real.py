#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Detector Node for QTrobot Confusion Detection System

This node interfaces with the QTrobot camera to detect faces
and extract facial features for confusion detection.
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from qt_confusion_detection.msg import FaceFeatures
import os

class FaceDetectorNode:
    """ROS node for face detection and feature extraction."""

    def __init__(self):
        """Initialize the face detector node."""
        rospy.init_node('face_detector_node', anonymous=True)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Parameters
        self.detection_frequency = rospy.get_param('~detection_frequency', 10)  # Hz
        self.face_detection_threshold = rospy.get_param('~face_detection_threshold', 0.7)
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/color/image_raw')
        self.simulate_confusion = rospy.get_param('~simulate_confusion', False)  # Default to not simulating confusion
        self.cascade_path = rospy.get_param('~cascade_path', '')  # Path to the cascade file

        # Load face detection model
        self.load_face_detection_model()

        # Publishers and subscribers
        self.face_features_pub = rospy.Publisher('/vision/face_features', FaceFeatures, queue_size=10)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)

        rospy.loginfo(f"Face detector node initialized, listening on {self.camera_topic}")
        rospy.loginfo(f"Confusion simulation is {'enabled' if self.simulate_confusion else 'disabled'}")

    def load_face_detection_model(self):
        """Load the face detection model."""
        try:
            rospy.loginfo("Initializing face detection")
            
            # Try to find the cascade file in common locations
            cascade_paths = [
                self.cascade_path,  # User-specified path
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',  # OpenCV 4
                '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',   # OpenCV 3
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')  # Local directory
            ]
            
            # Try each path until we find a valid one
            for path in cascade_paths:
                if path and os.path.exists(path):
                    rospy.loginfo(f"Loading cascade file from {path}")
                    self.face_detector = cv2.CascadeClassifier(path)
                    if not self.face_detector.empty():
                        self.face_detector_initialized = True
                        rospy.loginfo("Face detection initialized successfully")
                        return
            
            # If we couldn't find a valid cascade file, use a fallback approach
            rospy.logwarn("Could not find a valid cascade file. Using fallback detection.")
            self.face_detector_initialized = False
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize face detection: {e}")
            self.face_detector_initialized = False

    def image_callback(self, data):
        """Process incoming image data."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Detect faces
            faces = self.detect_faces(cv_image)

            # Process each detected face
            for face in faces:
                # Extract facial features
                features = self.extract_facial_features(cv_image, face)

                # Publish face features
                self.publish_face_features(features, data.header)

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def detect_faces(self, image):
        """Detect faces in the input image."""
        # If simulation is enabled, return a simulated face
        if self.simulate_confusion:
            height, width = image.shape[:2]
            return [(width // 4, height // 4, width // 2, height // 2)]
        
        # If we have a valid face detector, use it
        if self.face_detector_initialized:
            try:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # If faces were detected, return them
                if len(faces) > 0:
                    rospy.loginfo(f"Detected {len(faces)} faces")
                    return faces
            except Exception as e:
                rospy.logerr(f"Error in face detection: {e}")
        
        # If no faces were detected or face detection failed, return an empty list
        return []

    def extract_facial_features(self, image, face):
        """Extract facial features from the detected face."""
        x, y, w, h = face

        # Create a FaceFeatures message
        features = FaceFeatures()

        # Set bounding box
        features.bbox = [float(x), float(y), float(w), float(h)]

        # Set detection confidence (placeholder)
        features.detection_confidence = 0.9

        # In a real implementation, we would extract:
        # - Facial landmarks
        # - Action units
        # - Head pose
        # - Eye gaze

        # Placeholder for facial landmarks (would be extracted using a dedicated model)
        features.landmarks = []

        # If simulation is enabled, simulate high values for action units associated with confusion
        # Otherwise, use neutral values
        if self.simulate_confusion:
            # AU04 (Brow Lowerer), AU07 (Lid Tightener), AU14 (Dimpler), AU24 (Lip Pressor)
            features.action_units = [
                0.8,  # AU01 (Inner Brow Raiser) - high value
                0.7,  # AU02 (Outer Brow Raiser) - high value
                0.9,  # AU04 (Brow Lowerer) - very high value (strong indicator of confusion)
                0.1,  # AU05 (Upper Lid Raiser)
                0.0,  # AU06 (Cheek Raiser)
                0.8,  # AU07 (Lid Tightener) - high value (indicator of confusion)
                0.0,  # AU09 (Nose Wrinkler)
                0.0,  # AU10 (Upper Lip Raiser)
                0.0,  # AU12 (Lip Corner Puller)
                0.7,  # AU14 (Dimpler) - high value (indicator of confusion)
                0.0,  # AU15 (Lip Corner Depressor)
                0.0,  # AU17 (Chin Raiser)
                0.0,  # AU20 (Lip Stretcher)
                0.0,  # AU23 (Lip Tightener)
                0.8,  # AU24 (Lip Pressor) - high value (indicator of confusion)
                0.0,  # AU25 (Lips Part)
                0.0,  # AU26 (Jaw Drop)
                0.0   # AU45 (Blink)
            ]
        else:
            # Neutral values for action units (no confusion)
            features.action_units = [0.1] * 18

        # Placeholder for head pose (would be extracted using a dedicated model)
        # Simulate head tilt (common in confusion) or neutral pose
        features.head_pose = [0.3, 0.2, 0.4] if self.simulate_confusion else [0.0, 0.0, 0.0]

        # Placeholder for eye gaze (would be extracted using a dedicated model)
        # Simulate looking away (common in confusion) or neutral gaze
        if self.simulate_confusion:
            features.left_eye_gaze = [0.5, 0.3, 0.8]  # x, y, z
            features.right_eye_gaze = [0.5, 0.3, 0.8]  # x, y, z
        else:
            features.left_eye_gaze = [0.0, 0.0, 1.0]  # x, y, z (looking straight)
            features.right_eye_gaze = [0.0, 0.0, 1.0]  # x, y, z (looking straight)

        return features

    def publish_face_features(self, features, header):
        """Publish the extracted face features."""
        # Set the header
        features.header = header

        # Publish the message
        self.face_features_pub.publish(features)

    def run(self):
        """Run the face detector node."""
        rate = rospy.Rate(self.detection_frequency)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = FaceDetectorNode()
        detector.run()
    except rospy.ROSInterruptException:
        pass
