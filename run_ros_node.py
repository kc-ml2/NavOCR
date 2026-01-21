import rclpy
from ppdet.utils.cli import ArgsParser

from src.navocr_ros import PaddleNavOcrNode


def parse_args():
    """
    Parse command line arguments for both PaddleDetection and ROS2 Node.
    """
    parser = ArgsParser()
    # Arguments for PaddleDetection
    parser.add_argument("--output_dir", type=str, default="results/ros_result", help="Directory to save output images.")
    parser.add_argument("--draw_threshold", type=float, default=0.5, help="Confidence threshold for detection.")
    
    return parser.parse_args()

def main():
    FLAGS = parse_args()
    
    rclpy.init()
    
    node = PaddleNavOcrNode(FLAGS)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()