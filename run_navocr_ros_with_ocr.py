#!/usr/bin/env python3
"""
NavOCR with OCR Text Recognition
원본: run_navocr_ros.py
추가 기능: PaddleOCR로 텍스트 인식 + RViz 이미지 토픽 발행
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
import queue
import time

# Try to import PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("WARNING: PaddleOCR not installed. OCR will be disabled.")
    print("To install: pip install paddleocr")


class NavOCRWithOCRNode(Node):
    def __init__(self):
        super().__init__('navocr_with_ocr_node')

        # Parameters
        self.declare_parameter('model_path', '/home/sehyeon/workspace/src/NavOCR/model/nav_ocr_weight.pt')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('output_dir', '/home/sehyeon/workspace/src/NavOCR/results/ros_result_ocr')
        self.declare_parameter('ocr_language', 'korean')  # 'korean' for Korean+English mixed text
        self.declare_parameter('image_publish_rate', 2.0)  # Hz - limit RViz update rate
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.output_dir = self.get_parameter('output_dir').value
        ocr_lang = self.get_parameter('ocr_language').value
        self.image_publish_rate = self.get_parameter('image_publish_rate').value
        
        # OCR is ALWAYS enabled - no option to disable
        self.enable_ocr = True
        
        # Rate limiting for image publishing
        self.last_image_publish_time = self.get_clock().now()
        self.image_publish_interval = 1.0 / self.image_publish_rate  # seconds
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('YOLO model loaded successfully')
        
        # Initialize PaddleOCR (GPU-optimized for better accuracy) - ALWAYS ENABLED
        if not PADDLEOCR_AVAILABLE:
            self.get_logger().error('PaddleOCR not installed! Cannot run without OCR.')
            self.get_logger().error('To install: pip install paddleocr')
            raise RuntimeError('PaddleOCR is required but not installed!')
        
        self.get_logger().info(f'Initializing PaddleOCR (language: {ocr_lang})...')
        try:
            # RTX 4090 optimized - maximum performance!
            self.ocr = PaddleOCR(
                lang=ocr_lang,
                use_textline_orientation=True,  # Enable text orientation detection
                text_det_thresh=0.25,  # Lower for better recall (was 0.3)
                text_det_box_thresh=0.4,  # Lower for more detections (was 0.5)  
                text_recognition_batch_size=32  # Increased from 20 for RTX 4090
            )
            self.get_logger().info('✓ PaddleOCR initialized with RTX 4090 optimized settings!')
        except Exception as e:
            self.get_logger().error(f'✗ PaddleOCR initialization failed: {e}')
            self.get_logger().warn('   Trying with default parameters...')
            try:
                # Fallback: absolute minimal parameters
                self.ocr = PaddleOCR(lang=ocr_lang)
                self.get_logger().info('✓ PaddleOCR initialized with default settings')
            except Exception as e2:
                self.get_logger().error(f'✗ Default fallback also failed: {e2}')
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
        
        # Performance mode: No caching, immediate OCR
        # Remove all caching and queue mechanisms for maximum performance
        
        self.get_logger().info('='*60)
        self.get_logger().info('NavOCR with OCR Node Started!')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'OCR: ALWAYS ENABLED (Automatic Recognition)')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(f'Subscribing to: /camera/infra1/image_rect_raw')
        self.get_logger().info(f'Publishing detections to: /navocr/detections')
        self.get_logger().info(f'Publishing annotated image to: /navocr/annotated_image')
        self.get_logger().info('='*60)

    def perform_ocr_immediate(self, image_crop):
        """
        Perform OCR immediately (no caching, no queue - maximum performance)
        OCR is ALWAYS performed - no option to disable
        """
        if image_crop.size == 0:
            self.get_logger().warn('Empty image crop, returning default label')
            return "prominent_sign"
        
        try:
            # GPU에서는 더 큰 이미지로 OCR (정확도 향상)
            h, w = image_crop.shape[:2]
            max_size = 800  # GPU 사용 시 큰 이미지로 처리
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image_crop = cv2.resize(image_crop, (int(w*scale), int(h*scale)))
            
            # 이미지 전처리 (대비 향상)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_crop = cv2.equalizeHist(image_crop)  # Histogram equalization
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)
            
            # Run OCR
            ocr_start = time.time()
            results = self.ocr.ocr(image_crop)
            ocr_time = time.time() - ocr_start
            
            # Parse results with improved filtering
            recognized_text = None
            if results and len(results) > 0:
                result = results[0]
                
                # New API: dict format
                if isinstance(result, dict):
                    if 'rec_texts' in result and 'rec_scores' in result:
                        texts = []
                        for text, conf in zip(result['rec_texts'], result['rec_scores']):
                            if conf > 0.6:  # High confidence only
                                text = text.strip()
                                if len(text) > 0:
                                    texts.append(text)
                        if texts:
                            recognized_text = ' '.join(texts)
                            recognized_text = ' '.join(recognized_text.split())
                
                # Old API: list format
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
                self.get_logger().info(f'✅ OCR: "{recognized_text}" (took {ocr_time:.2f}s)')
                return recognized_text
            else:
                self.get_logger().warn(f'⚠️  OCR: No text found (took {ocr_time:.2f}s)')
                return "no_text_detected"
            
        except Exception as e:
            self.get_logger().error(f'❌ OCR ERROR: {e}')
            return "ocr_error"

    # DEPRECATED FUNCTIONS - All manual labeling removed, OCR is always automatic
    def process_ocr_queue(self):
        """DEPRECATED - removed for performance mode"""
        pass
    
    def republish_detection_with_ocr(self, detection_key, ocr_text):
        """DEPRECATED - removed for performance mode"""
        pass
    
    def queue_ocr_task(self, image_crop, detection_key, bbox):
        """DEPRECATED - removed for performance mode"""
        pass
    
    def get_detection_label(self, bbox, class_name="prominent_sign"):
        """DEPRECATED - removed for performance mode"""
        return class_name

    def perform_ocr_async(self, image_crop, detection_key, bbox):
        """DEPRECATED - removed threading to prevent segfault"""
        pass

    def perform_ocr(self, image_crop, detection_id):
        """DEPRECATED - Use perform_ocr_immediate instead"""
        return self.perform_ocr_immediate(image_crop)

    def draw_detection(self, image, x1, y1, x2, y2, label, conf):
        """
        Draw bounding box and label on image with Korean text support
        """
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Use ONLY the OCR label text (no confidence score)
        label_text = label  # Just the OCR text, no conf value
        
        # Check if label contains Korean characters
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in label)
        
        if has_korean:
            # Use PIL for Korean text rendering
            try:
                # Convert to PIL Image
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                # Try to load a Korean font (fallback to default if not found)
                try:
                    # Common Korean font paths on Ubuntu
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
                
                # Get text bounding box
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # Draw background rectangle
                draw.rectangle(
                    [(x1, y1 - text_h - 10), (x1 + text_w + 10, y1)],
                    fill=(0, 255, 0)
                )
                
                # Draw text
                draw.text((x1 + 5, y1 - text_h - 5), label_text, fill=(0, 0, 0), font=font)
                
                # Convert back to OpenCV format
                image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.get_logger().warn(f"Failed to draw Korean text with PIL: {e}")
                # Fallback to OpenCV (will show boxes for Korean chars)
                self._draw_opencv_text(image, x1, y1, label_text)
        else:
            # Use OpenCV for English text (faster)
            self._draw_opencv_text(image, x1, y1, label_text)
    
    def _draw_opencv_text(self, image, x1, y1, label_text):
        """Helper function to draw text with OpenCV (English only)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )

    def image_callback(self, msg):
        """
        Callback for image topic - PERFORMANCE MODE (No caching, immediate OCR)
        """
        # ROS Image → OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # YOLO inference
        results = self.model(cv_image, conf=self.conf_threshold)

        # Create annotated image (copy for drawing)
        annotated_image = cv_image.copy()

        # Create detection message
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        detection_count = 0
        
        # Process YOLO results
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Apply confidence threshold
                    if conf < self.conf_threshold:
                        continue
                    
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(cv_image.shape[1], x2)
                    y2 = min(cv_image.shape[0], y2)
                    
                    # Crop image for OCR
                    cropped_image = cv_image[y1:y2, x1:x2]
                    
                    detection_count += 1
                    
                    # **AUTOMATIC OCR** - Always performed, no manual labeling
                    if cropped_image.size > 0:
                        self.get_logger().info(f"🔍 [Detection {detection_count}] Running OCR...")
                        ocr_text = self.perform_ocr_immediate(cropped_image)
                        self.get_logger().info(f"✅ [Detection {detection_count}] OCR Result: '{ocr_text}'")
                    else:
                        self.get_logger().warn(f"⚠️  Empty crop (size={cropped_image.size})")
                        ocr_text = "empty_crop"
                    
                    # Skip detection if OCR failed (no valid text detected)
                    # Don't create markers for failed OCR results
                    skip_detection = ocr_text in ["no_text_detected", "empty_crop", "ocr_error"]
                    
                    if skip_detection:
                        self.get_logger().warn(f"⏭️  [Detection {detection_count}] Skipping - OCR failed: '{ocr_text}'")
                        # Still draw on image for debugging, but don't publish detection
                        self.draw_detection(annotated_image, x1, y1, x2, y2, f"SKIP: {ocr_text}", conf)
                        continue  # Skip this detection - don't add to detection_array
                    
                    # Draw detection on annotated image with OCR text as label
                    self.draw_detection(annotated_image, x1, y1, x2, y2, ocr_text, conf)
                    self.get_logger().info(f"🎨 [Detection {detection_count}] Drew label: '{ocr_text}' on image")
                    
                    # Create Detection2D message (only for successful OCR)
                    detection = Detection2D()
                    detection.header = msg.header
                    
                    # Bounding box
                    detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                    detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    
                    # Confidence with OCR text
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = ocr_text  # OCR text as class ID
                    hypothesis.hypothesis.score = conf
                    detection.results.append(hypothesis)
                    
                    detection_array.detections.append(detection)
                    
                    self.get_logger().info(
                        f"✅ Published detection: '{ocr_text}' conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})"
                    )
        
        # Publish detections (모든 OCR 완료된 상태로 발행!)
        self.detection_pub.publish(detection_array)
        self.get_logger().info(f"📤 Published {len(detection_array.detections)} detections to /navocr/detections")
        
        # Publish annotated image to RViz
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")
        
        # Save image to file (rate limit file saving to reduce disk I/O)
        self.frame_id += 1
        if self.frame_id % 10 == 0:  # Save every 10th frame only
            filename = os.path.join(self.output_dir, f"frame_{self.frame_id:06d}.png")
            cv2.imwrite(filename, annotated_image)
            
            # Log status (only when saving)
            if detection_count > 0:
                self.get_logger().info(
                    f"Frame {self.frame_id}: {detection_count} detections | Saved: {filename}"
                )


def main():
    rclpy.init()
    
    try:
        node = NavOCRWithOCRNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down NavOCR with OCR node...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
