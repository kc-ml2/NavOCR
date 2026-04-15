#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import os

import time

from navocr.detector import PaddleDetector

# PaddleOCR for text recognition
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("WARNING: PaddleOCR not installed. OCR will be disabled.")
    print("To install: pip install paddleocr")


class DetectorFlags:
    def __init__(self, config, weights, draw_threshold, output_dir):
        self.config = config
        self.weights = weights
        self.draw_threshold = draw_threshold
        self.output_dir = output_dir


def _get_navocr_root():
    """Resolve NavOCR root directory (where configs/, model/ etc. live).

    When installed via ament_python, __file__ points to site-packages/navocr/,
    which does not contain configs/ or model/. We check if the __file__-based
    path actually has configs/, otherwise search upward for the source tree.
    """
    # Try __file__-based resolution first (works when running from source)
    file_based = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(file_based, 'configs')):
        return file_based

    # Search upward from cwd for a NavOCR directory with configs/
    cwd = os.getcwd()
    for _ in range(5):
        candidate = os.path.join(cwd, 'src', 'NavOCR')
        if os.path.isdir(os.path.join(candidate, 'configs')):
            return candidate
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent

    # Last resort
    return file_based


class NavOCRWithOCRNode(Node):
    def __init__(self):
        super().__init__('navocr_with_ocr_node')

        # Resolve NavOCR root directory
        default_root = _get_navocr_root()

        # Parameters — navocr_root allows overriding the base path
        self.declare_parameter('navocr_root', default_root)
        navocr_root = self.get_parameter('navocr_root').value

        self.declare_parameter('config_path', os.path.join(navocr_root, 'configs/ppyoloe/ppyoloe_crn_s_infer_only.yml'))
        self.declare_parameter('weights_path', os.path.join(navocr_root, 'model/navocr.pdparams'))
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('output_dir', os.path.join(navocr_root, 'results/ros_result_ocr'))
        self.declare_parameter('ocr_language', 'en')
        self.declare_parameter('save_image', False)
        self.declare_parameter('benchmark', False)
        self.declare_parameter('session_name', '')

        # Get parameters
        config_path = self.get_parameter('config_path').value
        weights_path = self.get_parameter('weights_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.output_dir = self.get_parameter('output_dir').value
        ocr_lang = self.get_parameter('ocr_language').value
        self.save_image = self.get_parameter('save_image').value
        self.benchmark = self.get_parameter('benchmark').value
        self.session_name = self.get_parameter('session_name').value

        # Generate session name from timestamp if not provided
        if not self.session_name:
            from datetime import datetime
            self.session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.get_logger().info(f'Session name: {self.session_name}')
        self.get_logger().info(f'NavOCR root: {navocr_root}')

        # Create output directory
        if self.save_image or self.benchmark:
            os.makedirs(self.output_dir, exist_ok=True)

        # CV Bridge
        self.bridge = CvBridge()

        # Temporary file for PaddleDetector (requires file path input)
        self.temp_file_path = '/tmp/navocr_temp_frame.jpg'

        # Initialize PaddleDetector
        self.get_logger().info(f'Loading PaddleDetector...')
        self.get_logger().info(f'  Config: {config_path}')
        self.get_logger().info(f'  Weights: {weights_path}')

        flags = DetectorFlags(
            config=config_path,
            weights=weights_path,
            draw_threshold=self.conf_threshold,
            output_dir=self.output_dir
        )

        try:
            self.detector = PaddleDetector(flags)
            self.get_logger().info('PaddleDetector loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load PaddleDetector: {e}')
            raise RuntimeError(f'PaddleDetector initialization failed: {e}')

        # Initialize PaddleOCR
        if not PADDLEOCR_AVAILABLE:
            self.get_logger().error('PaddleOCR not installed! Cannot run without OCR.')
            raise RuntimeError('PaddleOCR is required but not installed!')

        self.get_logger().info(f'Initializing PaddleOCR (language: {ocr_lang})...')
        try:
            self.ocr = PaddleOCR(
                lang=ocr_lang,
                use_angle_cls=True,
                det_db_thresh=0.25,
                det_db_box_thresh=0.4,
                rec_batch_num=32
            )
            self.get_logger().info('PaddleOCR initialized successfully!')
        except Exception as e:
            self.get_logger().error(f'PaddleOCR initialization failed: {e}')
            self.get_logger().warn('Trying with default parameters...')
            try:
                self.ocr = PaddleOCR(lang=ocr_lang)
                self.get_logger().info('PaddleOCR initialized with default settings')
            except Exception as e2:
                raise RuntimeError(f'PaddleOCR initialization failed: {e2}')

        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image,
            "/camera/infra1/image_rect_raw",
            self.image_callback,
            10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/navocr/detections",
            10
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            "/navocr/annotated_image",
            10
        )

        if self.save_image:
            self.frame_id = 0

        # Drawing constants
        self.bbox_color = (0, 255, 0)
        self.bbox_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (0, 0, 0)

        # OCR constants
        self.ocr_conf_threshold = 0.6
        self.ocr_max_resize = 800
        self.image_save_interval = 10

        # OCR result sentinels
        self.OCR_NO_TEXT = "no_text_detected"
        self.OCR_ERROR = "ocr_error"
        self.OCR_FAIL_RESULTS = {self.OCR_NO_TEXT, "empty_crop", self.OCR_ERROR}

        # Performance metrics
        self.total_processing_time = 0.0
        self.frame_count = 0
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.total_ocr_time = 0.0
        self.wall_clock_start = None
        self.wall_clock_end = None

        self.get_logger().info('='*60)
        self.get_logger().info('NavOCR with OCR Node Started! (PaddleDetection + PaddleOCR)')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(f'Subscribing to: /camera/infra1/image_rect_raw')
        self.get_logger().info(f'Publishing detections to: /navocr/detections')
        self.get_logger().info(f'Publishing annotated image to: /navocr/annotated_image')
        self.get_logger().info('='*60)

    def perform_ocr_immediate(self, image_crop):
        if image_crop.size == 0:
            return self.OCR_NO_TEXT

        try:
            h, w = image_crop.shape[:2]
            if max(h, w) > self.ocr_max_resize:
                scale = self.ocr_max_resize / max(h, w)
                image_crop = cv2.resize(image_crop, (int(w*scale), int(h*scale)))

            # Preprocessing
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_crop = cv2.equalizeHist(image_crop)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)

            # Run OCR
            results = self.ocr.ocr(image_crop)

            # Parse results
            recognized_text = None
            if results and len(results) > 0:
                result = results[0]

                if isinstance(result, dict):
                    if 'rec_texts' in result and 'rec_scores' in result:
                        texts = []
                        for text, conf in zip(result['rec_texts'], result['rec_scores']):
                            if conf > self.ocr_conf_threshold:
                                text = text.strip()
                                if len(text) > 0:
                                    texts.append(text)
                        if texts:
                            recognized_text = ' '.join(texts)
                            recognized_text = ' '.join(recognized_text.split())

                elif isinstance(result, list) and len(result) > 0:
                    texts = []
                    for line in result:
                        if len(line) > 1:
                            text = line[1][0].strip()
                            conf = line[1][1]
                            if conf > self.ocr_conf_threshold and len(text) > 0:
                                texts.append(text)
                    if texts:
                        recognized_text = ' '.join(texts)
                        recognized_text = ' '.join(recognized_text.split())

            if recognized_text:
                return recognized_text
            else:
                return self.OCR_NO_TEXT

        except Exception as e:
            self.get_logger().error(f'OCR ERROR: {e}')
            return self.OCR_ERROR

    def draw_detection(self, image, x1, y1, x2, y2, label):
        cv2.rectangle(image, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
        self._draw_opencv_text(image, x1, y1, label)

    def _draw_opencv_text(self, image, x1, y1, label_text):
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, self.font, self.font_scale, self.font_thickness
        )

        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            self.bbox_color,
            -1
        )

        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 5),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness
        )

    def image_callback(self, msg):
        frame_start_time = time.time()

        if self.wall_clock_start is None:
            self.wall_clock_start = frame_start_time

        # ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Save to temp file for PaddleDetector
        cv2.imwrite(self.temp_file_path, cv_image)

        # PaddleDetection inference
        detection_start = time.time()
        results = self.detector.infer([self.temp_file_path])
        detection_time = time.time() - detection_start

        # Create annotated image
        annotated_image = cv_image.copy()

        # Create detection message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        frame_ocr_time = 0.0

        # Process detection results
        if results and 'bbox' in results[0]:
            for bbox in results[0]['bbox']:
                cls_id, score, x1, y1, x2, y2 = bbox

                # Apply confidence threshold
                if score < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(cv_image.shape[1], x2)
                y2 = min(cv_image.shape[0], y2)

                # Crop image for OCR
                cropped_image = cv_image[y1:y2, x1:x2]

                # Perform OCR
                if cropped_image.size > 0:
                    ocr_start = time.time()
                    ocr_text = self.perform_ocr_immediate(cropped_image)
                    ocr_time_single = time.time() - ocr_start
                    frame_ocr_time += ocr_time_single
                else:
                    ocr_text = self.OCR_NO_TEXT

                # Skip if OCR failed
                if ocr_text in self.OCR_FAIL_RESULTS:
                    self.draw_detection(annotated_image, x1, y1, x2, y2, "text")
                    continue

                # Draw detection with OCR text
                self.draw_detection(annotated_image, x1, y1, x2, y2, ocr_text)

                # Create Detection2D message
                detection = Detection2D()
                detection.header = msg.header

                # Bounding box
                detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                # Confidence with OCR text
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = ocr_text
                hypothesis.hypothesis.score = float(score)
                detection.results.append(hypothesis)

                detection_array.detections.append(detection)

                self.get_logger().info(
                    f"Detection: '{ocr_text}' conf={score:.2f} bbox=({x1},{y1},{x2},{y2})"
                )

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

        # Update performance metrics
        total_time = time.time() - frame_start_time
        self.total_processing_time += total_time
        self.frame_count += 1
        self.detection_count += len(detection_array.detections)
        self.total_detection_time += detection_time
        self.total_ocr_time += frame_ocr_time
        self.wall_clock_end = time.time()

        # Save image periodically if save_image mode is on
        if self.save_image:
            self.frame_id += 1
            if self.frame_id % self.image_save_interval == 0:
                filename = os.path.join(self.output_dir, f"frame_{self.frame_id:06d}.png")
                cv2.imwrite(filename, annotated_image)

                avg_time = self.total_processing_time / self.frame_count
                self.get_logger().info(
                    f"Frame {self.frame_id}: {len(detection_array.detections)} detections | "
                    f"Det: {detection_time:.3f}s | OCR: {frame_ocr_time:.3f}s | "
                    f"Total: {total_time:.3f}s | Avg: {avg_time:.3f}s"
                )

    def destroy_node(self):
        if self.frame_count > 0 and self.benchmark:
            avg_time = self.total_processing_time / self.frame_count
            avg_detection = self.total_detection_time / self.frame_count
            avg_ocr = self.total_ocr_time / self.frame_count
            avg_detections = self.detection_count / self.frame_count

            wall_clock_elapsed = 0.0
            if self.wall_clock_start is not None and self.wall_clock_end is not None:
                wall_clock_elapsed = self.wall_clock_end - self.wall_clock_start

            throughput_fps = self.frame_count / wall_clock_elapsed if wall_clock_elapsed > 0 else 0

            self.get_logger().info('='*60)
            self.get_logger().info('NavOCR (PaddleDetection+OCR) Final Statistics:')
            self.get_logger().info(f'  Total frames processed: {self.frame_count}')
            self.get_logger().info(f'  Total detections: {self.detection_count}')
            self.get_logger().info(f'  Avg detections/frame: {avg_detections:.2f}')
            self.get_logger().info(f'  Avg detection time: {avg_detection:.3f}s')
            self.get_logger().info(f'  Avg OCR time: {avg_ocr:.3f}s')
            self.get_logger().info(f'  Avg processing time per frame: {avg_time:.3f}s')
            self.get_logger().info(f'  Wall clock elapsed: {wall_clock_elapsed:.2f}s')
            self.get_logger().info(f'  Throughput FPS: {throughput_fps:.2f}')
            self.get_logger().info('='*60)

            # Save statistics to file
            timing_file = os.path.join(self.output_dir, f'{self.session_name}_timing.txt')
            try:
                with open(timing_file, 'w') as f:
                    f.write('=== NavOCR (PaddleDetection+OCR) Timing Statistics ===\n')
                    f.write(f'Session: {self.session_name}\n\n')
                    f.write(f'Total frames processed: {self.frame_count}\n')
                    f.write(f'Total detections: {self.detection_count}\n')
                    f.write(f'Avg detections/frame: {avg_detections:.2f}\n\n')
                    f.write(f'Total processing time: {self.total_processing_time:.3f}s\n')
                    f.write(f'Avg detection time: {avg_detection:.3f}s\n')
                    f.write(f'Avg OCR time: {avg_ocr:.3f}s\n')
                    f.write(f'Avg processing time per frame: {avg_time:.3f}s\n')
                    f.write(f'Wall clock elapsed: {wall_clock_elapsed:.2f}s\n')
                    f.write(f'Throughput FPS: {throughput_fps:.2f}\n')
                self.get_logger().info(f'Timing statistics saved to: {timing_file}')
            except Exception as e:
                self.get_logger().error(f'Failed to save timing statistics: {e}')

        # Clean up temp file
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

        super().destroy_node()


def main():
    rclpy.init()

    node = None
    try:
        node = NavOCRWithOCRNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down NavOCR with OCR node...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
