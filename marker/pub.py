#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import json

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Initialize subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
            
        # Get robot ID parameter
        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value
        
        # Initialize publisher
        self.detect_pub = self.create_publisher(
            String,
            f'robot_{self.robot_id}/object_detection',
            10)
            
        # Store latest depth image and odometry
        self.latest_depth = None
        self.latest_odom = None
        
        # Camera intrinsics (adjust for your camera)
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5

    def odom_callback(self, msg):
        self.latest_odom = msg

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg)

    def image_callback(self, msg):
        if self.latest_depth is None:
            return

        # Convert ROS image to CV2
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run detection
        results = self.model(cv_image)
        
        # Process results
        detected_objects = []
        
        for det in results.xyxy[0]:
            if det[-1] > 0.5:  # Confidence threshold
                x_mid = int((det[0] + det[2]) / 2)
                y_mid = int((det[1] + det[3]) / 2)
                
                # Get depth value
                z = self.latest_depth[y_mid, x_mid] / 1000.0  # Convert to meters
                
                # Calculate 3D coordinates
                x = (x_mid - self.cx) * z / self.fx
                y = (y_mid - self.cy) * z / self.fy
                
                # Create detection dictionary
                detection = {
                    'id': int(det[-1]),
                    'label': int(det[-1]),
                    'position': {'x': float(x), 'y': float(y), 'z': float(z)},
                    'robot_id': self.robot_id
                }
                
                detected_objects.append(detection)
        
        # Publish results as JSON string
        msg = String()
        msg.data = json.dumps(detected_objects)
        self.detect_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    detector = ObjectDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()