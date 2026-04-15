#!/usr/bin/env python3
"""
NavOCR with OpenVINO framework
(Imported from RT-DETR for text detection, and from PaddleOCR for text recognition)
"""

import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from navocr.detector_vino import OpenVINODetector


REC_H = 48
REC_IMG_W = 320
REC_MAX_W = 3200


class DetectorFlags:
    def __init__(self, config, weights, draw_threshold, output_dir, device='CPU', imgsz=640):
        self.config = config
        self.weights = weights
        self.draw_threshold = draw_threshold
        self.output_dir = output_dir
        self.device = device
        self.imgsz = imgsz
        self.det_model_path = weights


def _get_navocr_root():
    """Resolve NavOCR root directory (where configs/, model/ etc. live)."""
    file_based = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(file_based, 'configs')):
        return file_based

    cwd = os.getcwd()
    for _ in range(5):
        candidate = os.path.join(cwd, 'src', 'NavOCR')
        if os.path.isdir(os.path.join(candidate, 'configs')) or os.path.isdir(os.path.join(candidate, 'model')):
            return candidate
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent

    return file_based


def load_char_list(dict_path: str) -> list:
    p = Path(dict_path)
    if not p.exists():
        raise FileNotFoundError(f'Character dictionary not found: {dict_path}')

    if p.suffix.lower() in ('.yml', '.yaml'):
        import yaml
        with open(p, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        chars = cfg['PostProcess']['character_dict']
        return [''] + [str(c) for c in chars] + [' ']

    with open(p, encoding='utf-8') as f:
        chars = [line.rstrip('\n') for line in f]
    return [''] + chars + [' ']


def preprocess_rec(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    max_wh = max(REC_IMG_W / REC_H, w / h)
    img_w = min(int(REC_H * max_wh), REC_MAX_W)
    resized_w = min(img_w, int(math.ceil(REC_H * w / h)))
    resized = cv2.resize(crop_bgr, (resized_w, REC_H))
    img = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    img = (img - 0.5) / 0.5
    canvas = np.zeros((3, REC_H, img_w), dtype=np.float32)
    canvas[:, :, :resized_w] = img
    return canvas[np.newaxis]


def ctc_decode_with_conf(probs: np.ndarray, char_list: list):
    indices = np.argmax(probs, axis=-1)
    best = np.max(probs, axis=-1)
    result = []
    confs = []
    prev = -1
    for idx, score in zip(indices, best):
        idx = int(idx)
        if idx != prev:
            if idx != 0:
                result.append(char_list[idx])
                confs.append(float(score))
        prev = idx
    text = ''.join(result)
    conf = float(np.mean(confs)) if confs else 0.0
    return text, conf


class NavOCRWithOCRNode(Node):
    def __init__(self):
        super().__init__('navocr_with_ocr_node')

        default_root = _get_navocr_root()
        default_pd_config = os.path.join(default_root, 'configs/ppyoloe/ppyoloe_crn_s_infer_only.yml')
        default_pd_weights = os.path.join(default_root, 'model/navocr.pdparams')
        default_det_model = os.path.join(default_root, 'model/rtv4_hgnetv2_s.xml')
        default_rec_model = os.path.join(default_root, 'model/PP-OCRv5_mobile_rec_infer.xml')
        default_dict = os.path.join(default_root, 'model/PP-OCRv5_mobile_rec_infer', 'inference.yml')

        # Keep the original ROS2 parameter interface, and add OV-specific params.
        self.declare_parameter('navocr_root', default_root)
        navocr_root = self.get_parameter('navocr_root').value

        self.declare_parameter('config_path', default_pd_config)
        self.declare_parameter('weights_path', default_pd_weights)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('output_dir', os.path.join(navocr_root, 'results/ros_result_ocr'))
        self.declare_parameter('ocr_language', 'en')
        self.declare_parameter('save_image', False)
        self.declare_parameter('benchmark', False)
        self.declare_parameter('session_name', '')

        self.declare_parameter('det_model_path', default_det_model)
        self.declare_parameter('rec_model_path', default_rec_model)
        self.declare_parameter('dict_path', default_dict)
        self.declare_parameter('ov_device', 'CPU')
        self.declare_parameter('det_imgsz', 640)

        config_path = self.get_parameter('config_path').value
        weights_path = self.get_parameter('weights_path').value
        self.conf_threshold = float(self.get_parameter('confidence_threshold').value)
        self.output_dir = self.get_parameter('output_dir').value
        self.ocr_language = self.get_parameter('ocr_language').value
        self.save_image = bool(self.get_parameter('save_image').value)
        self.benchmark = bool(self.get_parameter('benchmark').value)
        self.session_name = self.get_parameter('session_name').value
        self.det_model_path = self.get_parameter('det_model_path').value
        self.rec_model_path = self.get_parameter('rec_model_path').value
        self.dict_path = self.get_parameter('dict_path').value
        self.ov_device = self.get_parameter('ov_device').value
        self.det_imgsz = int(self.get_parameter('det_imgsz').value)

        # Backward-compatibility fallback: if a user passes the old weights_path as an XML.
        if weights_path and weights_path.endswith('.xml') and self.det_model_path == default_det_model:
            self.det_model_path = weights_path

        if not self.session_name:
            self.session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.get_logger().info(f'Session name: {self.session_name}')
        self.get_logger().info(f'NavOCR root: {navocr_root}')
        self.get_logger().info(f'OpenVINO device: {self.ov_device}')
        self.get_logger().info(f'OpenVINO det model: {self.det_model_path}')
        self.get_logger().info(f'OpenVINO rec model: {self.rec_model_path}')
        self.get_logger().info(f'Char dict: {self.dict_path}')
        self.get_logger().info(f'Legacy config_path kept for compatibility: {config_path}')
        self.get_logger().info(f'Legacy ocr_language kept for compatibility: {self.ocr_language}')

        if self.save_image or self.benchmark:
            os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.temp_file_path = '/tmp/navocr_temp_frame.jpg'

        flags = DetectorFlags(
            config=config_path,
            weights=self.det_model_path,
            draw_threshold=self.conf_threshold,
            output_dir=self.output_dir,
            device=self.ov_device,
            imgsz=self.det_imgsz,
        )

        try:
            self.detector = OpenVINODetector(flags)
            self.get_logger().info('OpenVINO detector loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load OpenVINO detector: {e}')
            raise RuntimeError(f'OpenVINO detector initialization failed: {e}')

        try:
            import openvino as ov
            self.ov = ov
            core = ov.Core()
            config = {'PERFORMANCE_HINT': 'LATENCY'}
            if str(self.ov_device).upper().startswith('GPU'):
                config['INFERENCE_PRECISION_HINT'] = 'f32'
            self.rec_model = core.compile_model(core.read_model(self.rec_model_path), self.ov_device, config=config)
            self.char_list = load_char_list(self.dict_path)
            self.get_logger().info('OpenVINO recognizer loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load OpenVINO recognizer: {e}')
            raise RuntimeError(f'OpenVINO recognizer initialization failed: {e}')

        self.image_sub = self.create_subscription(
            Image,
            '/camera/infra1/image_rect_raw',
            self.image_callback,
            10,
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/navocr/detections',
            10,
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            '/navocr/annotated_image',
            10,
        )

        if self.save_image:
            self.frame_id = 0

        self.bbox_color = (0, 255, 0)
        self.bbox_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (0, 0, 0)

        self.ocr_conf_threshold = 0.6
        self.image_save_interval = 10
        self.OCR_NO_TEXT = 'no_text_detected'
        self.OCR_ERROR = 'ocr_error'
        self.OCR_FAIL_RESULTS = {self.OCR_NO_TEXT, 'empty_crop', self.OCR_ERROR}

        self.total_processing_time = 0.0
        self.frame_count = 0
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.total_ocr_time = 0.0
        self.wall_clock_start = None
        self.wall_clock_end = None

        self.get_logger().info('=' * 60)
        self.get_logger().info('NavOCR with OCR Node Started! (OpenVINO det + OpenVINO rec)')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info('Subscribing to: /camera/infra1/image_rect_raw')
        self.get_logger().info('Publishing detections to: /navocr/detections')
        self.get_logger().info('Publishing annotated image to: /navocr/annotated_image')
        self.get_logger().info('=' * 60)

    def perform_ocr_immediate(self, image_crop):
        """Perform OCR immediately on cropped image"""
        if image_crop.size == 0:
            return self.OCR_NO_TEXT

        try:
            rec_in = preprocess_rec(image_crop)
            rec_out = list(self.rec_model.infer_new_request({0: rec_in}).values())[0]
            text, conf = ctc_decode_with_conf(rec_out[0], self.char_list)
            text = ' '.join(text.strip().split())
            if text and conf >= self.ocr_conf_threshold:
                return text
            return self.OCR_NO_TEXT
        except Exception as e:
            self.get_logger().error(f'OCR ERROR: {e}')
            return self.OCR_ERROR

    def draw_detection(self, image, x1, y1, x2, y2, label):
        cv2.rectangle(image, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
        self._draw_opencv_text(image, x1, y1, label)

    def _draw_opencv_text(self, image, x1, y1, label_text):
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            self.font,
            self.font_scale,
            self.font_thickness,
        )

        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            self.bbox_color,
            -1,
        )

        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 5),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
        )

    def image_callback(self, msg):
        """Callback for image topic"""
        frame_start_time = time.time()

        if self.wall_clock_start is None:
            self.wall_clock_start = frame_start_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        cv2.imwrite(self.temp_file_path, cv_image)

        detection_start = time.time()
        results = self.detector.infer([self.temp_file_path])
        detection_time = time.time() - detection_start

        annotated_image = cv_image.copy()
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        frame_ocr_time = 0.0

        if results and 'bbox' in results[0]:
            for bbox in results[0]['bbox']:
                cls_id, score, x1, y1, x2, y2 = bbox
                del cls_id

                if float(score) < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(cv_image.shape[1], x2)
                y2 = min(cv_image.shape[0], y2)

                cropped_image = cv_image[y1:y2, x1:x2]
                if cropped_image.size > 0:
                    ocr_start = time.time()
                    ocr_text = self.perform_ocr_immediate(cropped_image)
                    frame_ocr_time += time.time() - ocr_start
                else:
                    ocr_text = self.OCR_NO_TEXT

                if ocr_text in self.OCR_FAIL_RESULTS:
                    self.draw_detection(annotated_image, x1, y1, x2, y2, 'text')
                    continue

                self.draw_detection(annotated_image, x1, y1, x2, y2, ocr_text)

                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = ocr_text
                hypothesis.hypothesis.score = float(score)
                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

                self.get_logger().info(
                    f"Detection: '{ocr_text}' conf={float(score):.2f} bbox=({x1},{y1},{x2},{y2})"
                )

        self.detection_pub.publish(detection_array)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

        total_time = time.time() - frame_start_time
        self.total_processing_time += total_time
        self.frame_count += 1
        self.detection_count += len(detection_array.detections)
        self.total_detection_time += detection_time
        self.total_ocr_time += frame_ocr_time
        self.wall_clock_end = time.time()

        if self.save_image:
            self.frame_id += 1
            if self.frame_id % self.image_save_interval == 0:
                filename = os.path.join(self.output_dir, f'frame_{self.frame_id:06d}.png')
                cv2.imwrite(filename, annotated_image)

                avg_time = self.total_processing_time / self.frame_count
                self.get_logger().info(
                    f'Frame {self.frame_id}: {len(detection_array.detections)} detections | '
                    f'Det: {detection_time:.3f}s | OCR: {frame_ocr_time:.3f}s | '
                    f'Total: {total_time:.3f}s | Avg: {avg_time:.3f}s'
                )

    def destroy_node(self):
        """Print final statistics on shutdown"""
        if self.frame_count > 0 and self.benchmark:
            avg_time = self.total_processing_time / self.frame_count
            avg_detection = self.total_detection_time / self.frame_count
            avg_ocr = self.total_ocr_time / self.frame_count
            avg_detections = self.detection_count / self.frame_count

            wall_clock_elapsed = 0.0
            if self.wall_clock_start is not None and self.wall_clock_end is not None:
                wall_clock_elapsed = self.wall_clock_end - self.wall_clock_start

            throughput_fps = self.frame_count / wall_clock_elapsed if wall_clock_elapsed > 0 else 0.0

            self.get_logger().info('=' * 60)
            self.get_logger().info('NavOCR (OpenVINO det + rec) Final Statistics:')
            self.get_logger().info(f'  Total frames processed: {self.frame_count}')
            self.get_logger().info(f'  Total detections: {self.detection_count}')
            self.get_logger().info(f'  Avg detections/frame: {avg_detections:.2f}')
            self.get_logger().info(f'  Avg detection time: {avg_detection:.3f}s')
            self.get_logger().info(f'  Avg OCR time: {avg_ocr:.3f}s')
            self.get_logger().info(f'  Avg processing time per frame: {avg_time:.3f}s')
            self.get_logger().info(f'  Wall clock elapsed: {wall_clock_elapsed:.2f}s')
            self.get_logger().info(f'  Throughput FPS: {throughput_fps:.2f}')
            self.get_logger().info('=' * 60)

            timing_file = os.path.join(self.output_dir, f'{self.session_name}_timing.txt')
            try:
                with open(timing_file, 'w', encoding='utf-8') as f:
                    f.write('=== NavOCR (OpenVINO det + rec) Timing Statistics ===\n')
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


if __name__ == '__main__':
    main()
