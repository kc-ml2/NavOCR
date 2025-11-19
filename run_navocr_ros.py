import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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