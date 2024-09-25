from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="/home/sooyong/datasets/yolo-dataset/textbox(2,2)/sim0.7_training_10k(7.5|2.5;6:2:2)/data.yaml", epochs=100, imgsz=640, batch=32, lr0=0.01, optimizer='Adam', save_period=10, device='0,1')
