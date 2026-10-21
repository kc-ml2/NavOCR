import cv2
import numpy as np

CLASS_COLORS = {
    0: (0, 255, 0),  # Green for class 0
    1: (0, 0, 255),  # Red for class 1
}

OCR_STATUS_TEXT = {
    'no_text_detected': 'No text recognized',
    'empty_crop': 'Empty crop',
    'ocr_error': 'OCR error',
}


def get_class_color(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_id, (0, 255, 0))  # Default to green if class_id not found


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
    text_x = max(0, min(x1, image.shape[1] - text_w - 6))
    text_bottom = min(image.shape[0] - baseline - 3, max(text_h + 3, y1 - 4))
    cv2.rectangle(
        image,
        (text_x, text_bottom - text_h - 3),
        (text_x + text_w + 6, text_bottom + baseline + 3),
        text_bg_color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label_text,
        (text_x + 3, text_bottom),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


def _wrap_text(text: str, max_width: int, font_scale: float, thickness: int) -> list[str]:
    """Wrap at spaces where possible, including text with no spaces."""
    lines = []
    for paragraph in text.splitlines() or ['']:
        line = ''
        for char in paragraph:
            width = cv2.getTextSize(line + char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
            if line and width > max_width:
                split_at = line.rfind(' ')
                if split_at > 0:
                    lines.append(line[:split_at])
                    line = line[split_at + 1:] + char
                else:
                    lines.append(line)
                    line = char
            else:
                line += char
        lines.append(line.strip())
    return lines


def _draw_ocr_panel(items: list[tuple[int, dict]], min_height: int, scale: float) -> np.ndarray:
    """Draw numbered class 0 OCR results on a separate panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 * scale
    thickness = max(1, round(scale))
    width, padding = round(480 * scale), round(24 * scale)
    line_height, row_gap = round(32 * scale), round(16 * scale)
    number_text = str(items[-1][0]) if items else '1'
    number_width = cv2.getTextSize(number_text, font, font_scale, thickness)[0][0]
    text_x = padding + number_width + padding
    rows = []
    for number, result in items:
        text = result['text'].strip()
        text = OCR_STATUS_TEXT.get(text, text) or 'No text recognized'
        lines = _wrap_text(text, width - text_x - padding, font_scale, thickness)
        rows.append((number, lines))

    y = round(144 * scale)
    content_height = sum(len(lines) * line_height + row_gap for _, lines in rows) or line_height
    panel = np.full((max(min_height, y + content_height + padding), width, 3), 248, dtype=np.uint8)
    class_name = items[0][1]['class_name'] if items else 'signboard_text'
    cv2.putText(panel, 'OCR RESULTS', (padding, round(48 * scale)), font, 0.85 * scale,
                (40, 40, 40), thickness, cv2.LINE_AA)
    cv2.putText(panel, f'{class_name} (class 0)', (padding, round(84 * scale)), font, 0.6 * scale,
                (100, 100, 100), thickness, cv2.LINE_AA)
    cv2.line(panel, (padding, round(104 * scale)), (width - padding, round(104 * scale)),
             (215, 215, 215), thickness)
    if not rows:
        cv2.putText(panel, 'No detections', (padding, y), font, font_scale,
                    (100, 100, 100), thickness, cv2.LINE_AA)
    for number, lines in rows:
        cv2.putText(panel, str(number), (padding, y), font, font_scale,
                    (0, 130, 0), thickness, cv2.LINE_AA)
        for line in lines:
            cv2.putText(panel, line, (text_x, y), font, font_scale,
                        (40, 40, 40), thickness, cv2.LINE_AA)
            y += line_height
        y += row_gap
    return panel


def render_result_image(orig_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """Render class labels and an OCR panel without modifying the inputs."""
    vis = orig_bgr.copy()
    height, width = vis.shape[:2]
    scale = max(1.0, min(height, width) / 1200)
    ordered_results = sorted(results, key=lambda result: result['class_id'])
    numbered_results = list(enumerate(ordered_results, start=1))
    for number, result in numbered_results:
        label = f"{number} {result['class_name']}"
        draw_detection(
            vis, *result['box'], label,
            bbox_color=get_class_color(result['class_id']),
            bbox_thickness=max(2, round(2 * scale)),
            font_scale=0.65 * scale,
            font_thickness=max(1, round(scale)),
        )

    class0_items = [(number, result) for number, result in numbered_results if result['class_id'] == 0]
    panel = _draw_ocr_panel(class0_items, height, scale)
    canvas = np.full((panel.shape[0], width + panel.shape[1], 3), 248, dtype=np.uint8)
    canvas[:height, :width] = vis
    canvas[:, width:] = panel
    return canvas


def clip_bbox(image, x1, y1, x2, y2) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return x1, y1, x2, y2


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    return image
