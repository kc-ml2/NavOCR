from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

from .manifest_io import DetectionRow, ManifestIO, ManifestRow, PipelineConfig

logger = logging.getLogger(__name__)

SplitRatios = tuple[float, float, float]


class COCOExporter:
    CATEGORY: dict = {"id": 1, "name": "navigation-oriented-text"}
    DEFAULT_SPLIT: SplitRatios = (0.70, 0.15, 0.15)

    def __init__(
        self,
        config: PipelineConfig,
        split_ratios: SplitRatios = DEFAULT_SPLIT,
    ) -> None:
        self._config = config
        self._split_ratios = split_ratios

    def assign_splits(self, rows: list[ManifestRow]) -> dict[str, str]:
        preserved: dict[str, str] = {}
        unassigned: list[ManifestRow] = []
        for row in rows:
            if row.split in ("train", "val", "test"):
                preserved[row.image_filename] = row.split
            else:
                unassigned.append(row)

        store_to_images: dict[str, list[str]] = defaultdict(list)
        for row in unassigned:
            store_to_images[row.store_name].append(row.image_filename)

        store_lists = list(store_to_images.values())
        store_indices = [0] * len(store_lists)
        ordered: list[str] = []

        while True:
            added_any = False
            for i, images in enumerate(store_lists):
                idx = store_indices[i]
                if idx < len(images):
                    ordered.append(images[idx])
                    store_indices[i] += 1
                    added_any = True
            if not added_any:
                break

        total = len(ordered)
        train_ratio, val_ratio, _ = self._split_ratios
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        result: dict[str, str] = dict(preserved)
        for i, filename in enumerate(ordered):
            if i < train_end:
                result[filename] = "train"
            elif i < val_end:
                result[filename] = "val"
            else:
                result[filename] = "test"

        return result

    def polygon_to_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        x4: float,
        y4: float,
    ) -> list[float]:
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        min_x = min(xs)
        min_y = min(ys)
        width = max(xs) - min_x
        height = max(ys) - min_y
        return [min_x, min_y, width, height]

    def run(
        self,
        rows: list[ManifestRow],
        detection_rows: list[DetectionRow],
        image_dir: Path,
        output_root: Path,
        manifest_io: ManifestIO,
    ) -> None:
        ocr_filtered_rows = [r for r in rows if r.status in ("ocr_filtered", "exported")]

        split_map = self.assign_splits(ocr_filtered_rows)

        for split in ("train", "val", "test"):
            (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "annotations").mkdir(parents=True, exist_ok=True)

        detections_by_image: dict[str, list[DetectionRow]] = defaultdict(list)
        for dr in detection_rows:
            if dr.is_prominent_sign:
                detections_by_image[dr.image_filename].append(dr)

        coco: dict[str, dict] = {
            split: {
                "images": [],
                "annotations": [],
                "categories": [self.CATEGORY],
            }
            for split in ("train", "val", "test")
        }

        image_id_counter = 0
        annotation_id_counter = 0

        for row in ocr_filtered_rows:
            filename = row.image_filename
            split = split_map.get(filename, "train")
            try:
                src_path = image_dir / filename
                dst_path = output_root / "images" / split / filename
                shutil.copy2(src_path, dst_path)

                with Image.open(src_path) as img:
                    width, height = img.size

                image_id_counter += 1
                image_id = image_id_counter

                coco[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": width,
                        "height": height,
                    }
                )

                prominent_detections = detections_by_image.get(filename, [])
                for dr in prominent_detections:
                    bbox = self.polygon_to_bbox(
                        dr.x1, dr.y1,
                        dr.x2, dr.y2,
                        dr.x3, dr.y3,
                        dr.x4, dr.y4,
                    )
                    area = bbox[2] * bbox[3]
                    annotation_id_counter += 1
                    coco[split]["annotations"].append(
                        {
                            "id": annotation_id_counter,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )

                manifest_io.update_row(
                    filename,
                    split=split,
                    status="exported",
                    num_label_boxes=len(prominent_detections),
                )
            except Exception:
                logger.exception("Export failed for %s", filename)
                manifest_io.update_row(filename, status="error")

        for split in ("train", "val", "test"):
            annotation_path = output_root / "annotations" / f"instances_{split}.json"
            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(coco[split], f, indent=2, ensure_ascii=False)
