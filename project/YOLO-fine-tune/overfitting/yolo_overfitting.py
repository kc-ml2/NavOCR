from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="/home/sooyong/datasets/yolo-dataset/sim0.7_overfitting_10k(7.5|2.5)/data.yaml",
            epochs=200, imgsz=640, batch=16, lr0=0.001, optimizer='SGD', augment=False, save_period=10, device='0,1')
