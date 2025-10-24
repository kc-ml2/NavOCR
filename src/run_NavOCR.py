import csv
import os
import time

from ultralytics import YOLO

# Load our YOLO model trained for NavOCR
our_model = YOLO("./model/nav_ocr_weight.pt")

# Load key paths and confidence threshold
INPUT_ROOT = os.environ.get("INPUT_ROOT")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT")
CONFIDENCE = 0.5

# Run inference
results = our_model(
    source=f"{INPUT_ROOT}/images",
    save=True,
    show=False,
    project=OUTPUT_ROOT,
    name="NavOCR_images",
    stream=True,
    conf=CONFIDENCE,
)


def save_to_csv(file_path, filename, x1, y1, x2, y2, conf):
    """Save filtered keyframes"""

    try:
        file_path = os.path.expanduser(file_path)
        file_exist = os.path.exists(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exist:
                header = ["image_filename", "x1", "y1", "x2", "y2", "conf"]
                writer.writerow(header)
            writer.writerow([filename, x1, y1, x2, y2, conf])
    except Exception as e:
        print(f"CSV write error: {e}")


def run_yolo(results):
    """Run NavOCR and save results"""

    for r in results:
        if not r.boxes:
            continue
        file_name = os.path.basename(r.path)
        for box in r.boxes.data:
            x1, y1, x2, y2, conf = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            save_to_csv(f"{OUTPUT_ROOT}/NavOCR_info.csv", file_name, x1, y1, x2, y2, conf)


def main():
    """Run NavOCR and print inference time."""
    start = time.time()
    run_yolo(results)
    print(f"Execution time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
