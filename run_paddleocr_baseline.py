#!/usr/bin/env python3
"""
PaddleOCR Baseline Node - For Benchmark Comparison
NavOCR (YOLO + PaddleOCR) vs Pure PaddleOCR comparison

This node runs PaddleOCR directly on the full image without YOLO detection.
It detects text regions and recognizes text in a single pass.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
import time

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("ERROR: PaddleOCR not installed!")
    print("To install: pip install paddleocr")


class PaddleOCRBaselineNode(Node):
    """
    Pure PaddleOCR baseline for benchmark comparison.

    Key differences from NavOCR:
    - No YOLO detection stage
    - PaddleOCR handles both text detection AND recognition
    - Processes full image directly
    """

    def __init__(self):
        super().__init__('paddleocr_baseline_node')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.3)  # OCR confidence threshold (lowered from 0.6)
        self.declare_parameter('output_dir', '/home/sehyeon/ros2_ws/src/NavOCR/results/paddleocr_baseline')
        self.declare_parameter('ocr_language', 'korean')
        self.declare_parameter('min_text_length', 2)  # Minimum text length to consider
        self.declare_parameter('min_box_area', 500)   # Minimum bounding box area (pixels^2)

        # Get parameters
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.output_dir = self.get_parameter('output_dir').value
        ocr_lang = self.get_parameter('ocr_language').value
        self.min_text_length = self.get_parameter('min_text_length').value
        self.min_box_area = self.get_parameter('min_box_area').value

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # CV Bridge
        self.bridge = CvBridge()

        # Initialize PaddleOCR
        if not PADDLEOCR_AVAILABLE:
            self.get_logger().error('PaddleOCR not installed!')
            raise RuntimeError('PaddleOCR is required but not installed!')

        self.get_logger().info(f'Initializing PaddleOCR (language: {ocr_lang})...')
        try:
            # RTX 4090 optimized settings - same as NavOCR for fair comparison
            self.ocr = PaddleOCR(
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
            self.ocr = PaddleOCR(lang=ocr_lang)
            self.get_logger().info('PaddleOCR initialized with default settings')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/camera/infra1/image_rect_raw",
            self.image_callback,
            10
        )

        # Publishers - same topics as NavOCR for compatibility
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/paddleocr_baseline/detections",  # Different topic to avoid conflict
            10
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            "/paddleocr_baseline/annotated_image",
            10
        )

        self.frame_id = 0
        self.first_frame_logged = False  # For one-time image info logging

        # Performance metrics
        self.total_processing_time = 0.0
        self.frame_count = 0
        self.detection_count = 0

        self.get_logger().info('='*60)
        self.get_logger().info('PaddleOCR Baseline Node Started!')
        self.get_logger().info('This is for benchmark comparison with NavOCR')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'Min text length: {self.min_text_length}')
        self.get_logger().info(f'Min box area: {self.min_box_area}')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(f'Subscribing to: /camera/infra1/image_rect_raw')
        self.get_logger().info(f'Publishing detections to: /paddleocr_baseline/detections')
        self.get_logger().info('='*60)

    def polygon_to_bbox(self, polygon):
        """
        Convert polygon points to bounding box (x1, y1, x2, y2)
        PaddleOCR returns 4-point polygons for text regions
        """
        points = np.array(polygon)
        x1 = int(np.min(points[:, 0]))
        y1 = int(np.min(points[:, 1]))
        x2 = int(np.max(points[:, 0]))
        y2 = int(np.max(points[:, 1]))
        return x1, y1, x2, y2

    def draw_detection(self, image, x1, y1, x2, y2, label, conf):
        """Draw bounding box and label with Korean text support"""
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for PaddleOCR baseline

        label_text = f"{label}"

        # Check for Korean characters
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in label)

        if has_korean:
            try:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

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

                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                # Blue background for PaddleOCR baseline (to distinguish from NavOCR's green)
                draw.rectangle(
                    [(x1, y1 - text_h - 10), (x1 + text_w + 10, y1)],
                    fill=(255, 0, 0)
                )
                draw.text((x1 + 5, y1 - text_h - 5), label_text, fill=(255, 255, 255), font=font)

                image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.get_logger().warn(f"Failed to draw Korean text: {e}")
                self._draw_opencv_text(image, x1, y1, label_text)
        else:
            self._draw_opencv_text(image, x1, y1, label_text)

    def _draw_opencv_text(self, image, x1, y1, label_text):
        """Draw text with OpenCV (English only)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Blue background
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    def image_callback(self, msg):
        """Process image with pure PaddleOCR (no YOLO)"""
        start_time = time.time()

        # ROS Image -> OpenCV
        # Handle both color and grayscale (infrared) images
        try:
            # Log image info on first frame
            if not self.first_frame_logged:
                self.get_logger().info(f"First frame received: encoding={msg.encoding}, "
                                       f"size={msg.width}x{msg.height}")
                self.first_frame_logged = True

            # First try passthrough to get original format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Check if grayscale (infrared image is mono8 or mono16)
            if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
                # Grayscale image - apply preprocessing for better OCR
                # Normalize if mono16
                if cv_image.dtype == np.uint16:
                    cv_image = (cv_image / 256).astype(np.uint8)

                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                # This improves contrast for infrared images which often have low contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cv_image = clahe.apply(cv_image)

                # Convert to BGR for PaddleOCR
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                self.get_logger().debug(f"Preprocessed infrared image: {cv_image.shape}")
            elif cv_image.shape[2] == 3:
                # Already BGR or RGB
                pass

        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        annotated_image = cv_image.copy()

        # Create detection message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # Run PaddleOCR on full image (detection + recognition in one pass)
        ocr_start = time.time()
        try:
            results = self.ocr.ocr(cv_image)
        except Exception as e:
            self.get_logger().error(f"PaddleOCR failed: {e}")
            return
        ocr_time = time.time() - ocr_start

        frame_detections = 0
        raw_detections = 0  # Count before filtering

        # Process OCR results
        if results and len(results) > 0:
            result = results[0]

            # Log raw OCR result structure on first few frames for debugging
            if self.frame_id < 3:
                self.get_logger().info(f"Frame {self.frame_id}: OCR result type={type(result)}, "
                                       f"result={result}")

            if result is not None:
                # Handle different PaddleOCR API formats
                if isinstance(result, dict):
                    # New API format
                    if 'rec_texts' in result and 'rec_scores' in result and 'det_polygons' in result:
                        polygons = result['det_polygons']
                        texts = result['rec_texts']
                        scores = result['rec_scores']
                        raw_detections = len(texts)

                        for polygon, text, score in zip(polygons, texts, scores):
                            if self._process_detection(
                                polygon, text, score,
                                msg.header, detection_array, annotated_image
                            ):
                                frame_detections += 1

                elif isinstance(result, list):
                    # Old API format: list of [polygon, (text, confidence)]
                    raw_detections = len(result)
                    for line in result:
                        if len(line) >= 2:
                            polygon = line[0]
                            text = line[1][0]
                            score = line[1][1]

                            if self._process_detection(
                                polygon, text, score,
                                msg.header, detection_array, annotated_image
                            ):
                                frame_detections += 1

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

        # Update metrics
        total_time = time.time() - start_time
        self.total_processing_time += total_time
        self.frame_count += 1
        self.detection_count += len(detection_array.detections)

        # Save frame periodically
        self.frame_id += 1
        if self.frame_id % 10 == 0:
            filename = os.path.join(self.output_dir, f"frame_{self.frame_id:06d}.png")
            cv2.imwrite(filename, annotated_image)

            avg_time = self.total_processing_time / self.frame_count
            self.get_logger().info(
                f"Frame {self.frame_id}: {len(detection_array.detections)}/{raw_detections} detections (after/before filter) | "
                f"OCR: {ocr_time:.3f}s | Total: {total_time:.3f}s | Avg: {avg_time:.3f}s"
            )

    def _process_detection(self, polygon, text, score, header, detection_array, annotated_image):
        """Process a single detection and add to array if valid. Returns True if added."""
        # Filter by confidence
        if score < self.conf_threshold:
            self.get_logger().debug(f"Filtered by confidence: '{text}' conf={score:.2f} < {self.conf_threshold}")
            return False

        # Filter by text length
        text = text.strip()
        if len(text) < self.min_text_length:
            self.get_logger().debug(f"Filtered by text length: '{text}' len={len(text)} < {self.min_text_length}")
            return False

        # Convert polygon to bbox
        x1, y1, x2, y2 = self.polygon_to_bbox(polygon)

        # Filter by box area
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_box_area:
            self.get_logger().debug(f"Filtered by box area: '{text}' area={area} < {self.min_box_area}")
            return False

        # Draw on annotated image
        self.draw_detection(annotated_image, x1, y1, x2, y2, text, score)

        # Create Detection2D message
        detection = Detection2D()
        detection.header = header

        # Bounding box
        detection.bbox.center.position.x = float((x1 + x2) / 2.0)
        detection.bbox.center.position.y = float((y1 + y2) / 2.0)
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)

        # Hypothesis with OCR text
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = text
        hypothesis.hypothesis.score = float(score)
        detection.results.append(hypothesis)

        detection_array.detections.append(detection)

        self.get_logger().info(
            f"Valid detection: '{text}' conf={score:.2f} bbox=({x1},{y1},{x2},{y2})"
        )
        return True

    def destroy_node(self):
        """Print final statistics on shutdown and save to file"""
        if self.frame_count > 0:
            avg_time = self.total_processing_time / self.frame_count
            avg_detections = self.detection_count / self.frame_count

            # Print to terminal
            self.get_logger().info('='*60)
            self.get_logger().info('PaddleOCR Baseline Final Statistics:')
            self.get_logger().info(f'  Total frames: {self.frame_count}')
            self.get_logger().info(f'  Total detections: {self.detection_count}')
            self.get_logger().info(f'  Avg detections/frame: {avg_detections:.2f}')
            self.get_logger().info(f'  Avg processing time: {avg_time:.3f}s')
            self.get_logger().info(f'  Avg FPS: {1.0/avg_time:.2f}')
            self.get_logger().info('='*60)

            # Save to file
            timing_file = os.path.join(self.output_dir, 'timing_statistics.txt')
            try:
                with open(timing_file, 'w') as f:
                    f.write('=== PaddleOCR Baseline Timing Statistics ===\n')
                    f.write(f'Total frames: {self.frame_count}\n')
                    f.write(f'Total detections: {self.detection_count}\n')
                    f.write(f'Avg detections/frame: {avg_detections:.2f}\n')
                    f.write(f'Total processing time: {self.total_processing_time:.3f}s\n')
                    f.write(f'Avg processing time: {avg_time:.3f}s\n')
                    f.write(f'Avg FPS: {1.0/avg_time:.2f}\n')
                self.get_logger().info(f'Timing statistics saved to: {timing_file}')
            except Exception as e:
                self.get_logger().error(f'Failed to save timing statistics: {e}')

        super().destroy_node()


def main():
    rclpy.init()

    try:
        node = PaddleOCRBaselineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down PaddleOCR Baseline node...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
