from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data="/home/sooyong/datasets/yolo-dataset/7:1.5:1.5/zeta-10k/textbox(1,1)/data.yaml",
            epochs=150, imgsz=640, batch=64, optimizer='Adam', save_period=10, device='0,1')
