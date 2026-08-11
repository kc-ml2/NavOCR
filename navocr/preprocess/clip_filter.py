from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from .manifest_io import ManifestIO, ManifestRow, PipelineConfig

# Cascading 3-pass CLIP filter. Each pass is (negative_prompt, positive_prompt).
# An image passes the filter only if its positive-class probability is >=
# threshold for ALL three pass. The reported clip_score is the minimum across
# the three passes (= the weakest pass), which preserves the AND semantics in
# a single number: min(scores) >= threshold iff all three passes >= threshold.
PROMPT_SETS: list[tuple[str, str]] = [
    (
        "a photo that are not contain signboard or a photo of a signboard against a plain background",
        "a photo of a store exterior",
    ),
    (
        "a photo that are not contain signboard or a photo of a signboard against a plain background",
        "a photo of a store exterior with a signboard and surrounding environment",
    ),
    (
        "a online logo of a brand",
        "a photo of a store exterior with a signboard and surrounding environment",
    ),
]

logger = logging.getLogger(__name__)


class CLIPFilter:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._model: Optional[object] = None
        self._processor: Optional[object] = None

    def _load_model(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._model.to(self._config.device)

    def _score_batch(self, images: list[Image.Image]) -> list[float]:
        import torch

        per_pass_scores: list[list[float]] = []
        for negative, positive in PROMPT_SETS:
            inputs = self._processor(
                text=[negative, positive],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._config.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            per_pass_scores.append([float(probs[i][1].item()) for i in range(len(images))])
        return [min(s) for s in zip(*per_pass_scores)]

    def score_image(self, image_path: Path) -> float:
        try:
            if self._model is None:
                self._load_model()
            image = Image.open(image_path).convert("RGB")
            return self._score_batch([image])[0]
        except Exception:
            logger.exception("score_image failed for %s", image_path)
            return 0.0

    def run(
        self,
        rows: list[ManifestRow],
        image_dir: Path,
        manifest_io: ManifestIO,
    ) -> list[ManifestRow]:
        if self._model is None:
            self._load_model()

        pending = [r for r in rows if r.status == "pending"]
        batch_size = 32

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            images: list[Image.Image] = []
            valid_indices: list[int] = []
            error_indices: list[int] = []

            for idx, row in enumerate(batch):
                image_path = image_dir / row.image_filename
                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except Exception:
                    logger.exception("Failed to open image %s", image_path)
                    error_indices.append(idx)

            for idx in error_indices:
                row = batch[idx]
                row.status = "error"
                manifest_io.update_row(row.image_filename, status="error")

            if not images:
                continue

            try:
                min_scores = self._score_batch(images)
            except Exception:
                logger.exception("Batch CLIP inference failed")
                for row_idx in valid_indices:
                    row = batch[row_idx]
                    row.status = "error"
                    manifest_io.update_row(row.image_filename, status="error")
                continue

            for batch_img_idx, row_idx in enumerate(valid_indices):
                row = batch[row_idx]
                score = min_scores[batch_img_idx]
                row.clip_score = score
                row.status = "clip_pass" if score >= self._config.clip_threshold else "clip_fail"
                manifest_io.update_row(
                    row.image_filename,
                    clip_score=score,
                    status=row.status,
                )

        return rows
