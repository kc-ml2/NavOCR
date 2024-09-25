from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="/home/sooyong/datasets/yolo-dataset-zeta/textbox(1,1)/data.yaml",
            epochs=50, imgsz=640, batch=16, lr0=0.01, optimizer='Adam', save_period=10, device='0,1')
