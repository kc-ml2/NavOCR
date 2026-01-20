# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Chaehyeuk Lee (KC ML2) on 2026-01-14 to integrate with ROS2.


import time
import os
import glob

import ast
from ppdet.utils.cli import ArgsParser
from ppdet.utils.logger import setup_logger

from src.navocr import PaddleDetector


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--infer_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--draw_threshold", type=float, default=0.5, help="Drawing threshold.")
    parser.add_argument("--save_results", type=ast.literal_eval, default=False, help="Save JSON results.")
    parser.add_argument("--visualize", type=ast.literal_eval, default=False, help="Save visualized images.")
    parser.add_argument("--exts", type=str, default="jpg,jpeg,png,bmp", help="Image extensions.")
    return parser.parse_args()


def list_images(infer_dir, exts_csv):
    exts = [e.strip().lower() for e in exts_csv.split(",") if e.strip()]
    images = []
    for e in exts:
        images.extend(glob.glob(os.path.join(infer_dir, f"*.{e}")))
        images.extend(glob.glob(os.path.join(infer_dir, f"*.{e.upper()}")))
    return sorted(set(images))


def main():
    FLAGS = parse_args()
    logger = setup_logger("infer_test")

    detector = PaddleDetector(FLAGS)
    
    images = list_images(os.path.abspath(FLAGS.infer_dir), FLAGS.exts)
    if not images:
        logger.error("No images found!")
        return

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    logger.info("Starting memory-based inference loop with FPS measurement...")
    
    total_time = 0
    count = 0

    for img_path in images:
        start_time = time.time()
        
        results = detector.infer([img_path], visualize=FLAGS.visualize, save_results=FLAGS.save_results)
        
        end_time = time.time()
        duration = end_time - start_time
        total_time += duration
        count += 1
        fps = 1 / duration
        
        if 'bbox' in results[0]:
            bboxes = results[0]['bbox']
            logger.info(f"Image: {os.path.basename(img_path)} | Detected: {len(bboxes)} boxes | Time: {duration:.4f}s | FPS: {fps:.2f}")
            
            # # (Optional) Print bbox data
            # for box in bboxes:
            #     cls_id, score, x1, y1, x2, y2 = box
            #     if score > FLAGS.draw_threshold:
            #         print(f" >> Class ID: {int(cls_id)}, Score: {score:.4f}")
        else:
            logger.info(f"Image: {os.path.basename(img_path)} | No boxes | Time: {duration:.4f}s | FPS: {fps:.2f}")

    if count > 0:
        avg_fps = count / total_time
        logger.info("-" * 50)
        logger.info(f"Total Images: {count}")
        logger.info(f"Average Time per image: {total_time/count:.4f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
