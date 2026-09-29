from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import textdistance as td

from .manifest_io import DetectionIO, DetectionRow, ManifestIO, ManifestRow, PipelineConfig

try:
    import deepl as _deepl_lib
except ImportError as _deepl_import_error:
    _deepl_lib = None
    _DEEPL_IMPORT_ERROR = _deepl_import_error
else:
    _DEEPL_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class TranslationCache:
    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._data: dict[str, dict[str, str]] = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, text: str, target_lang: str) -> Optional[str]:
        return self._data.get(text, {}).get(target_lang)

    def set(self, text: str, target_lang: str, translation: str) -> None:
        if text not in self._data:
            self._data[text] = {}
        self._data[text][target_lang] = translation

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class Matcher:
    def __init__(self, config: PipelineConfig, cache: TranslationCache) -> None:
        self._config = config
        self._cache = cache

    def translate(self, text: str, target_lang: str) -> Optional[str]:
        if self._config.deepl_auth_key is None:
            return None
        cached = self._cache.get(text, target_lang)
        if cached is not None:
            return cached
        if _deepl_lib is None:
            logger.warning("deepl package is not installed; cannot translate. " "Install it with: pip install deepl")
            return None
        try:
            translator = _deepl_lib.Translator(self._config.deepl_auth_key)
            result = translator.translate_text(text, target_lang=target_lang)
            translation: str = result.text  # type: ignore[union-attr]
            self._cache.set(text, target_lang, translation)
            self._cache.flush()
            return translation
        except Exception as exc:
            logger.warning("DeepL translation failed for %r: %s", text, exc)
            return None

    def _compute_score(self, ocr_norm: str, store_name: str) -> float:
        ocr_lower = ocr_norm.lower()
        scores: list[float] = []
        for word in store_name.split():
            scores.append(td.levenshtein.normalized_similarity(word.lower(), ocr_lower))
        combined = store_name.replace(" ", "").lower()
        scores.append(td.levenshtein.normalized_similarity(combined, ocr_lower))
        return max(scores) if scores else 0.0

    def best_score(
        self,
        ocr_norm: str,
        store_name: str,
        translated_en: Optional[str],
        translated_ko: Optional[str],
    ) -> tuple[float, str]:
        score_original = self._compute_score(ocr_norm, store_name)
        best: float = score_original
        match_type: str = "original"

        if translated_en is not None:
            s = self._compute_score(ocr_norm, translated_en)
            if s > best:
                best = s
                match_type = "translated_en"

        if translated_ko is not None:
            s = self._compute_score(ocr_norm, translated_ko)
            if s > best:
                best = s
                match_type = "translated_ko"

        return best, match_type

    def run(
        self,
        rows: list[ManifestRow],
        manifest_io: ManifestIO,
        detection_io: DetectionIO,
    ) -> None:
        ocr_done = [r for r in rows if r.status == "ocr_done"]
        if not ocr_done:
            return

        all_detections = detection_io.read_all()
        detections_by_image: dict[str, list[DetectionRow]] = {}
        for det in all_detections:
            detections_by_image.setdefault(det.image_filename, []).append(det)

        for row in ocr_done:
            try:
                translated_en: Optional[str] = None
                translated_ko: Optional[str] = None

                lang = row.language
                if lang == "ko":
                    translated_en = self.translate(row.store_name, "EN-US")
                elif lang == "en":
                    translated_ko = self.translate(row.store_name, "KO")
                elif lang == "mixed":
                    translated_en = self.translate(row.store_name, "EN-US")
                    translated_ko = self.translate(row.store_name, "KO")

                image_dets = detections_by_image.get(row.image_filename, [])

                for det in image_dets:
                    score, mtype = self.best_score(
                        det.ocr_text_normalized,
                        row.store_name,
                        translated_en,
                        translated_ko,
                    )
                    det.levenshtein_score = score
                    det.match_type = mtype
                    det.is_prominent_sign = score >= self._config.similarity_threshold

                prominent = [d for d in image_dets if d.is_prominent_sign]
                num_label_boxes = len(prominent)

                top_ocr_text: Optional[str] = None
                best_lev_score: Optional[float] = None
                if prominent:
                    top_det = max(
                        prominent,
                        key=lambda d: d.levenshtein_score or 0.0,
                    )
                    top_ocr_text = top_det.ocr_text
                    best_lev_score = top_det.levenshtein_score

                translated_store_name: Optional[str] = translated_en or translated_ko

                manifest_io.update_row(
                    row.image_filename,
                    status="matched",
                    num_label_boxes=num_label_boxes,
                    top_ocr_text=top_ocr_text,
                    best_levenshtein_score=best_lev_score,
                    translated_store_name=translated_store_name,
                )

            except Exception as exc:
                logger.exception("Matcher error for image %r: %s", row.image_filename, exc)
                manifest_io.update_row(row.image_filename, status="error")

        if ocr_done:
            detection_io.write_all(all_detections)
