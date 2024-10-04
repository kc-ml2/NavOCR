from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="/home/sooyong/datasets/yolo-dataset/7:1.5:1.5/epsilon-20k/textbox(3,1.5)/data.yaml",
            epochs=150, imgsz=640, batch=32, optimizer='Adam', save_period=10, device='0')
