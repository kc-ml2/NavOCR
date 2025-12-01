import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO


class YoloInferNode(Node):
    def __init__(self):
        super().__init__('yolov8_inference_node')

        # Load custom YOLO model
        self.model = YOLO("./model/nav_ocr_weight.pt")

        self.bridge = CvBridge()

        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image,
            "/camera/infra1/image_rect_raw",
            self.image_callback,
            10
        )
        
        # Publisher for detection results
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/navocr/detections",
            10
        )

        self.frame_id = 0
        self.get_logger().info("YOLO inference node started (python script).")

    def image_callback(self, msg):
        # ROS Image → OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # YOLO inference
        results = self.model(cv_image)

        # Create detection message
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Create Detection2D message
                    detection = Detection2D()
                    detection.header = msg.header
                    
                    # Bounding box
                    detection.bbox.center.position.x = float((xyxy[0] + xyxy[2]) / 2.0)
                    detection.bbox.center.position.y = float((xyxy[1] + xyxy[3]) / 2.0)
                    detection.bbox.size_x = float(xyxy[2] - xyxy[0])
                    detection.bbox.size_y = float(xyxy[3] - xyxy[1])
                    
                    # Confidence
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = "text"
                    hypothesis.hypothesis.score = conf
                    detection.results.append(hypothesis)
                    
                    detection_array.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detection_array)
        self.get_logger().info(f"Published {len(detection_array.detections)} detections")

        # Bounding box 그리기
        annotated = results[0].plot()

        # Save image
        self.frame_id += 1
        filename = f"results/ros_result/frame_{self.frame_id:06d}.png"
        cv2.imwrite(filename, annotated)
        self.get_logger().info(f"Saved: {filename}")


def main():
    rclpy.init()
    node = YoloInferNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()