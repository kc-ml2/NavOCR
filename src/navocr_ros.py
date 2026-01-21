import os
import time

import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import warnings

from src.navocr import PaddleDetector


class PaddleNavOcrNode(Node):
    def __init__(self, flags):
        """
        Initialize the ROS2 Node and the PaddleDetection model.
        """
        super().__init__('paddle_navocr_node')
        
        self.get_logger().info("Initializing PaddleDetector...")        
        self.detector = PaddleDetector(flags)
    
        self.draw_threshold = flags.draw_threshold
        self.output_dir = flags.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Temporary file path to pass to the existing PaddleDetector.infer()
        self.temp_file_path = "temp_ros_inference.jpg"
        self.bridge = CvBridge()

        # Subscriber for the raw image topic
        self.subscription = self.create_subscription(
            Image,
            "/camera/infra1/image_rect_raw",
            self.image_callback,
            10
        )

        self.frame_id = 0
        self.get_logger().info("PaddleDetection ROS2 node started successfully with Temp-File bridge.")

    def image_callback(self, msg):
        """
        Callback function for the image subscription.
        Saves a temporary file to satisfy the file-path requirement of the detector.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Save current frame to a temporary file
        cv2.imwrite(self.temp_file_path, cv_image)

        start_time = time.perf_counter()
        results = self.detector.infer([self.temp_file_path])
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        fps = 1 / duration if duration > 0 else 0

        # Draw Bounding Boxes on the original BGR image
        self.frame_id += 1
        if results and 'bbox' in results[0]:
            bboxes = results[0]['bbox']
            self.get_logger().info(f"Frame {self.frame_id} | Detected: {len(bboxes)} boxes | FPS: {fps:.2f}")

            for box in bboxes:
                cls_id, score, x1, y1, x2, y2 = box
                if score > self.draw_threshold:
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID:{int(cls_id)} {score:.2f}"
                    cv2.putText(cv_image, label, (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            self.get_logger().info(f"Frame {self.frame_id} | No boxes detected | FPS: {fps:.2f}")

        # Save the final annotated image for verification
        filename = f"{self.output_dir}/frame_{self.frame_id:06d}.png"
        cv2.imwrite(filename, cv_image)
        
        # Periodic status logging
        if self.frame_id % 30 == 0:
            self.get_logger().info(f"Last saved frame: {filename}")