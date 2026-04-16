from __future__ import annotations

import cv2


def draw_detection(
    image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label_text: str,
    bbox_color=(0, 255, 0),
    bbox_thickness: int = 2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_color=(0, 0, 0),
    text_bg_color=None,
) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

    if text_bg_color is None:
        text_bg_color = bbox_color

    (text_w, text_h), baseline = cv2.getTextSize(
        label_text,
        font,
        font_scale,
        font_thickness,
    )
    text_bottom = max(text_h + baseline + 2, y1 - 4)
    cv2.rectangle(
        image,
        (x1, text_bottom - text_h - baseline),
        (x1 + text_w, text_bottom + baseline),
        text_bg_color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label_text,
        (x1, text_bottom),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


def clip_bbox(image, x1, y1, x2, y2) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return x1, y1, x2, y2
