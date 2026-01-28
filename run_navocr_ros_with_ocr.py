#!/usr/bin/env python3
"""
NavOCR with OCR Text Recognition (PaddleDetection + PaddleOCR)
Migrated from ultralytics YOLO to PaddleDetection (PPYOLOe)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import os
import sys
from PIL import Image as PILImage, ImageDraw, ImageFont
import time

# Add NavOCR to path for PaddleDetector import
navocr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, navocr_path)

from src.navocr import PaddleDetector

# PaddleOCR for text recognition
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("WARNING: PaddleOCR not installed. OCR will be disabled.")
    print("To install: pip install paddleocr")


class DetectorFlags:
    """Configuration flags for PaddleDetector"""
    def __init__(self, config, weights, draw_threshold, output_dir):
        self.config = config
        self.weights = weights
        self.draw_threshold = draw_threshold
        self.output_dir = output_dir


class NavOCRWithOCRNode(Node):
    def __init__(self):
        super().__init__('navocr_with_ocr_node')

        # Parameters
        self.declare_parameter('config_path', os.path.join(navocr_path, 'configs/ppyoloe/ppyoloe_crn_s_infer_only.yml'))
        self.declare_parameter('weights_path', os.path.join(navocr_path, 'model/navocr.pdparams'))
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('output_dir', os.path.join(navocr_path, 'results/ros_result_ocr'))
        self.declare_parameter('ocr_language', 'en')
        self.declare_parameter('image_publish_rate', 2.0)
        self.declare_parameter('session_name', '')

        # Get parameters
        config_path = self.get_parameter('config_path').value
        weights_path = self.get_parameter('weights_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.output_dir = self.get_parameter('output_dir').value
        ocr_lang = self.get_parameter('ocr_language').value
        self.image_publish_rate = self.get_parameter('image_publish_rate').value
        self.session_name = self.get_parameter('session_name').value

        # Generate session name from timestamp if not provided
        if not self.session_name:
            from datetime import datetime
            self.session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.get_logger().info(f'Session name: {self.session_name}')

        # Rate limiting for image publishing
        self.last_image_publish_time = self.get_clock().now()
        self.image_publish_interval = 1.0 / self.image_publish_rate

        # Create output directory
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
                use_gpu=True,
                lang=ocr_lang,
                use_textline_orientation=True,
                text_det_thresh=0.25,
                text_det_box_thresh=0.4,
                text_recognition_batch_size=32
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

        self.frame_id = 0

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
        """Perform OCR immediately on cropped image"""
        if image_crop.size == 0:
            return "no_text_detected"

        try:
            h, w = image_crop.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
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
                            if conf > 0.6:
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
                            if conf > 0.6 and len(text) > 0:
                                texts.append(text)
                    if texts:
                        recognized_text = ' '.join(texts)
                        recognized_text = ' '.join(recognized_text.split())

            if recognized_text:
                return recognized_text
            else:
                return "no_text_detected"

        except Exception as e:
            self.get_logger().error(f'OCR ERROR: {e}')
            return "ocr_error"

    def draw_detection(self, image, x1, y1, x2, y2, label, conf):
        """Draw bounding box and label on image with Korean text support"""
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = label

        has_korean = any('\uac00' <= char <= '\ud7a3' for char in label)

        if has_korean:
            try:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                try:
                    font_paths = [
                        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                    ]
                    font = None
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            font = ImageFont.truetype(font_path, 20)
                            break
                    if font is None:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                draw.rectangle(
                    [(x1, y1 - text_h - 10), (x1 + text_w + 10, y1)],
                    fill=(0, 255, 0)
                )
                draw.text((x1 + 5, y1 - text_h - 5), label_text, fill=(0, 0, 0), font=font)

                image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self._draw_opencv_text(image, x1, y1, label_text)
        else:
            self._draw_opencv_text(image, x1, y1, label_text)

    def _draw_opencv_text(self, image, x1, y1, label_text):
        """Helper function to draw text with OpenCV"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    def image_callback(self, msg):
        """Callback for image topic"""
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

        detection_count = 0
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

                detection_count += 1

                # Perform OCR
                if cropped_image.size > 0:
                    ocr_start = time.time()
                    ocr_text = self.perform_ocr_immediate(cropped_image)
                    ocr_time_single = time.time() - ocr_start
                    frame_ocr_time += ocr_time_single
                else:
                    ocr_text = "no_text_detected"

                # Skip if OCR failed
                if ocr_text in ["no_text_detected", "empty_crop", "ocr_error"]:
                    self.draw_detection(annotated_image, x1, y1, x2, y2, "text", score)
                    continue

                # Draw detection with OCR text
                self.draw_detection(annotated_image, x1, y1, x2, y2, ocr_text, score)

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

        # Save image periodically
        self.frame_id += 1
        if self.frame_id % 10 == 0:
            filename = os.path.join(self.output_dir, f"frame_{self.frame_id:06d}.png")
            cv2.imwrite(filename, annotated_image)

            avg_time = self.total_processing_time / self.frame_count
            self.get_logger().info(
                f"Frame {self.frame_id}: {detection_count} detections | "
                f"Det: {detection_time:.3f}s | OCR: {frame_ocr_time:.3f}s | "
                f"Total: {total_time:.3f}s | Avg: {avg_time:.3f}s"
            )

    def destroy_node(self):
        """Print final statistics on shutdown"""
        if self.frame_count > 0:
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
