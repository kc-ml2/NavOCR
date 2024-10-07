import csv
import os
from ultralytics import YOLO

model = YOLO("/home/sooyong/datasets/yolo-dataset/results/epsilon-15k_textbox(2,2)/train/weights/best.pt")
# # model = YOLO("yolov8n.pt")

# results = model("/home/sooyong/datasets/yolo-dataset/7:1.5:1.5/epsilon-15k/textbox(2,2)/images/test", save=True, show=True, project="/home/sooyong/datasets/test")
# results = model(source=0, save=True, show=True)
results = model(source="/home/sooyong/datasets/-- result --/coex-img", save=True, show=True, project="/home/sooyong/datasets/output-3")
# results = model(source="/home/sooyong/datasets/003016_1.png", save=False, show=True, project="output")

def save_to_csv(file_path, filename, x1, y1, x2, y2, conf):
    try:
        # 홈 디렉토리(~)를 절대 경로로 확장
        file_path = os.path.expanduser(file_path)
        
        file_exist = os.path.exists(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exist:
                header = ['image_filename', 'x1', 'y1', 'x2', 'y2', 'conf']
                writer.writerow(header)
            writer.writerow([filename, x1, y1, x2, y2, conf])
    except Exception as e:
        print(f"CSV 저장 오류: {e}")

# YOLO 결과를 CSV 파일에 저장하는 함수
def run_yolo(results):
    for r in results:
        if not r.boxes:
            continue
        else:
            file_name = os.path.basename(r.path)
            # 첫 번째 박스만 저장하는 예시 (여러 박스를 저장하고 싶다면 반복문 추가)
            for box in r.boxes.data:
                x1, y1, x2, y2, conf = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
                # 홈 디렉토리에 CSV 파일을 저장하도록 변경
                save_to_csv('/home/sooyong/datasets/output-3/yolo_info.csv', file_name, x1, y1, x2, y2, conf)

# YOLO 결과 처리 함수 실행
run_yolo(results)