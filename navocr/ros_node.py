#!/usr/bin/env python3

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from navocr.backend_factory import create_detector, create_ocr
from navocr.config_loader import load_detector_config, load_ocr_config
from navocr.ocr_base import BaseOCR
from navocr.pipeline_utils import clip_bbox, draw_detection


@dataclass
class ROSNodeConfig:
    params_file: str
    save_image: bool
    benchmark: bool
    session_name: str
    image_topic: str
    detections_topic: str
    annotated_image_topic: str
    queue_size: int
    temp_file_path: str
    image_save_interval: int

def get_navocr_share_dir() -> str:
    return get_package_share_directory('navocr')


class NavOCRNode(Node):
    def __init__(self):
        super().__init__('navocr_with_ocr_node')
        self.node_config = self._declare_parameters()
        # Detector/OCR model settings are loaded from the shared params YAML so
        # the same configuration can be reused by both ROS and standalone entry points.
        self.detector_config = load_detector_config(self.node_config.params_file)
        self.ocr_config = load_ocr_config(self.node_config.params_file)
        self.bridge = CvBridge()
        self.fallback_label = 'text'
        self.ocr_fail_results = {BaseOCR.NO_TEXT, 'empty_crop', BaseOCR.ERROR}

        self.total_processing_time = 0.0
        self.frame_count = 0
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.total_ocr_time = 0.0
        self.wall_clock_start = None
        self.wall_clock_end = None

        if self.node_config.save_image or self.node_config.benchmark:
            os.makedirs(self.detector_config.output_dir, exist_ok=True)
        if self.node_config.save_image:
            self.frame_id = 0

        self.detector = create_detector(self.detector_config)
        self.ocr = create_ocr(self.ocr_config)
        self._create_ros_io()
        self._log_startup()

    def _declare_parameters(self) -> ROSNodeConfig:
        default_share = get_navocr_share_dir()
        # These parameters control ROS node runtime behavior.
        # Detector/OCR model settings are loaded separately from `params_file`.
        self.declare_parameter('params_file', os.path.join(default_share, 'configs/navocr_openvino.params.yaml'))
        self.declare_parameter('save_image', False)
        self.declare_parameter('benchmark', False)
        self.declare_parameter('session_name', '')
        self.declare_parameter('image_topic', '/camera/infra1/image_rect_raw')
        self.declare_parameter('detections_topic', '/navocr/detections')
        self.declare_parameter('annotated_image_topic', '/navocr/annotated_image')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('temp_file_path', '/tmp/navocr_temp_frame.jpg')
        self.declare_parameter('image_save_interval', 10)

        session_name = self.get_parameter('session_name').value
        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        return ROSNodeConfig(
            params_file=self.get_parameter('params_file').value,
            save_image=bool(self.get_parameter('save_image').value),
            benchmark=bool(self.get_parameter('benchmark').value),
            session_name=session_name,
            image_topic=self.get_parameter('image_topic').value,
            detections_topic=self.get_parameter('detections_topic').value,
            annotated_image_topic=self.get_parameter('annotated_image_topic').value,
            queue_size=int(self.get_parameter('queue_size').value),
            temp_file_path=self.get_parameter('temp_file_path').value,
            image_save_interval=int(self.get_parameter('image_save_interval').value),
        )

    def _create_ros_io(self) -> None:
        self.image_sub = self.create_subscription(
            Image,
            self.node_config.image_topic,
            self.image_callback,
            self.node_config.queue_size,
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.node_config.detections_topic,
            self.node_config.queue_size,
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            self.node_config.annotated_image_topic,
            self.node_config.queue_size,
        )

    def _log_startup(self) -> None:
        self.get_logger().info(f'Session name: {self.node_config.session_name}')
        self.get_logger().info(f'Params file: {self.node_config.params_file}')
        self.get_logger().info(f'Detector backend: {self.detector_config.backend}')
        self.get_logger().info(f'OCR backend: {self.ocr_config.backend}')
        self.get_logger().info(f'Detector device: {self.detector_config.device or "auto"}')
        self.get_logger().info(f'OCR device: {self.ocr_config.device or "auto"}')
        self.get_logger().info(f'Detector threshold: {self.detector_config.detection_threshold}')
        self.get_logger().info(f'OCR threshold: {self.ocr_config.confidence_threshold}')
        self.get_logger().info(f'Output directory: {self.detector_config.output_dir}')
        self.get_logger().info(f'Subscribing to: {self.node_config.image_topic}')
        self.get_logger().info(f'Publishing detections to: {self.node_config.detections_topic}')
        self.get_logger().info(f'Publishing annotated image to: {self.node_config.annotated_image_topic}')

    def image_callback(self, msg) -> None:
        frame_start_time = time.time()
        if self.wall_clock_start is None:
            self.wall_clock_start = frame_start_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        detection_start = time.time()
        results = self.detector.infer_loaded_images([cv_image])
        if results is None:
            cv2.imwrite(self.node_config.temp_file_path, cv_image)
            results = self.detector.infer([self.node_config.temp_file_path])
        detection_time = time.time() - detection_start

        annotated_image = cv_image.copy()
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        frame_ocr_time = 0.0

        if results and 'bbox' in results[0]:
            for bbox in results[0]['bbox']:
                _cls_id, score, x1, y1, x2, y2 = bbox
                if float(score) < self.detector_config.detection_threshold:
                    continue

                x1, y1, x2, y2 = clip_bbox(cv_image, x1, y1, x2, y2)
                cropped_image = cv_image[y1:y2, x1:x2]
                ocr_text, frame_ocr_time = self._recognize_text(cropped_image, frame_ocr_time)

                if ocr_text in self.ocr_fail_results:
                    draw_detection(annotated_image, x1, y1, x2, y2, self.fallback_label)
                    continue

                draw_detection(annotated_image, x1, y1, x2, y2, ocr_text)
                detection_array.detections.append(
                    self._build_detection(msg.header, x1, y1, x2, y2, ocr_text, score)
                )
                self.get_logger().info(
                    f"Detection: '{ocr_text}' conf={float(score):.2f} bbox=({x1},{y1},{x2},{y2})"
                )

        self.detection_pub.publish(detection_array)
        self._publish_annotated_image(msg.header, annotated_image)
        total_time = self._update_performance_stats(
            frame_start_time,
            detection_time,
            frame_ocr_time,
            len(detection_array.detections),
        )
        self._save_frame_if_needed(
            annotated_image,
            len(detection_array.detections),
            detection_time,
            frame_ocr_time,
            total_time,
        )

    def _recognize_text(self, cropped_image, frame_ocr_time: float) -> tuple[str, float]:
        if cropped_image.size == 0:
            return BaseOCR.NO_TEXT, frame_ocr_time

        ocr_start = time.time()
        ocr_text = self.ocr.recognize(cropped_image)
        return ocr_text, frame_ocr_time + (time.time() - ocr_start)

    def _build_detection(self, header, x1: int, y1: int, x2: int, y2: int, ocr_text: str, score: float) -> Detection2D:
        detection = Detection2D()
        detection.header = header
        detection.bbox.center.position.x = float((x1 + x2) / 2.0)
        detection.bbox.center.position.y = float((y1 + y2) / 2.0)
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = ocr_text
        hypothesis.hypothesis.score = float(score)
        detection.results.append(hypothesis)
        return detection

    def _publish_annotated_image(self, header, annotated_image) -> None:
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish annotated image: {exc}')

    def _update_performance_stats(
        self,
        frame_start_time: float,
        detection_time: float,
        frame_ocr_time: float,
        detection_count: int,
    ) -> float:
        total_time = time.time() - frame_start_time
        self.total_processing_time += total_time
        self.frame_count += 1
        self.detection_count += detection_count
        self.total_detection_time += detection_time
        self.total_ocr_time += frame_ocr_time
        self.wall_clock_end = time.time()
        return total_time

    def _save_frame_if_needed(
        self,
        annotated_image,
        detection_count: int,
        detection_time: float,
        frame_ocr_time: float,
        total_time: float,
    ) -> None:
        if not self.node_config.save_image:
            return

        self.frame_id += 1
        if self.frame_id % self.node_config.image_save_interval != 0:
            return

        filename = os.path.join(self.detector_config.output_dir, f'frame_{self.frame_id:06d}.png')
        cv2.imwrite(filename, annotated_image)
        avg_time = self.total_processing_time / self.frame_count
        self.get_logger().info(
            f'Frame {self.frame_id}: {detection_count} detections | Det: {detection_time:.3f}s | '
            f'OCR: {frame_ocr_time:.3f}s | Total: {total_time:.3f}s | Avg: {avg_time:.3f}s'
        )

    def destroy_node(self):
        if self.frame_count > 0 and self.node_config.benchmark:
            self._log_final_statistics()
            self._write_timing_file()

        if os.path.exists(self.node_config.temp_file_path):
            os.remove(self.node_config.temp_file_path)

        super().destroy_node()

    def _log_final_statistics(self) -> None:
        avg_time = self.total_processing_time / self.frame_count
        avg_detection = self.total_detection_time / self.frame_count
        avg_ocr = self.total_ocr_time / self.frame_count
        avg_detections = self.detection_count / self.frame_count
        wall_clock_elapsed = 0.0
        if self.wall_clock_start is not None and self.wall_clock_end is not None:
            wall_clock_elapsed = self.wall_clock_end - self.wall_clock_start
        throughput_fps = self.frame_count / wall_clock_elapsed if wall_clock_elapsed > 0 else 0.0

        self.get_logger().info('=' * 60)
        self.get_logger().info('NavOCR Final Statistics:')
        self.get_logger().info(f'  Total frames processed: {self.frame_count}')
        self.get_logger().info(f'  Total detections: {self.detection_count}')
        self.get_logger().info(f'  Avg detections/frame: {avg_detections:.2f}')
        self.get_logger().info(f'  Avg detection time: {avg_detection:.3f}s')
        self.get_logger().info(f'  Avg OCR time: {avg_ocr:.3f}s')
        self.get_logger().info(f'  Avg processing time per frame: {avg_time:.3f}s')
        self.get_logger().info(f'  Wall clock elapsed: {wall_clock_elapsed:.2f}s')
        self.get_logger().info(f'  Throughput FPS: {throughput_fps:.2f}')
        self.get_logger().info('=' * 60)

    def _write_timing_file(self) -> None:
        avg_time = self.total_processing_time / self.frame_count
        avg_detection = self.total_detection_time / self.frame_count
        avg_ocr = self.total_ocr_time / self.frame_count
        avg_detections = self.detection_count / self.frame_count
        wall_clock_elapsed = 0.0
        if self.wall_clock_start is not None and self.wall_clock_end is not None:
            wall_clock_elapsed = self.wall_clock_end - self.wall_clock_start
        throughput_fps = self.frame_count / wall_clock_elapsed if wall_clock_elapsed > 0 else 0.0

        timing_file = os.path.join(self.detector_config.output_dir, f'{self.node_config.session_name}_timing.txt')
        try:
            with open(timing_file, 'w', encoding='utf-8') as handle:
                handle.write('=== NavOCR Timing Statistics ===\n')
                handle.write(f'Session: {self.node_config.session_name}\n\n')
                handle.write(f'Total frames processed: {self.frame_count}\n')
                handle.write(f'Total detections: {self.detection_count}\n')
                handle.write(f'Avg detections/frame: {avg_detections:.2f}\n\n')
                handle.write(f'Total processing time: {self.total_processing_time:.3f}s\n')
                handle.write(f'Avg detection time: {avg_detection:.3f}s\n')
                handle.write(f'Avg OCR time: {avg_ocr:.3f}s\n')
                handle.write(f'Avg processing time per frame: {avg_time:.3f}s\n')
                handle.write(f'Wall clock elapsed: {wall_clock_elapsed:.2f}s\n')
                handle.write(f'Throughput FPS: {throughput_fps:.2f}\n')
            self.get_logger().info(f'Timing statistics saved to: {timing_file}')
        except Exception as exc:
            self.get_logger().error(f'Failed to save timing statistics: {exc}')


def main():
    rclpy.init()

    node = None
    try:
        node = NavOCRNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down NavOCR with OCR node...')
    except Exception as exc:
        print(f'Error: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
