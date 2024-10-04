# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")

# model.train(data="/home/sooyong/datasets/yolo-dataset/7:1.5:1.5/zeta-10k/textbox(2,2)/data.yaml",
#             epochs=150, imgsz=640, batch=64, optimizer='SGD', save_period=10, device='0,1')


from ultralytics import YOLO
import itertools
import os

# 설정할 파라미터 리스트
# size_list = ["18k", "15k"]
size_list = ["10k"]
textbox_list = ["(1,1)"]

combinations = list(itertools.product(size_list, textbox_list))

for size, textbox in combinations:
    # 데이터 경로 설정
    data_path = f"/home/sooyong/datasets/yolo-dataset/7:1.5:1.5/zeta-{size}/textbox{textbox}/data.yaml"

    # 결과를 저장할 디렉토리 설정
    save_dir = f"/home/sooyong/datasets/yolo-dataset/results/zeta-{size}_textbox{textbox}"
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO("yolo11n.pt")

    # 모델 학습
    model.train(
        data=data_path,
        epochs=150,
        imgsz=640,
        batch=64,
        optimizer='SGD',
        save_period=10,
        device='0',
        project=save_dir
    )

    print(f"Training completed for size: {size}, textbox: {textbox}")
