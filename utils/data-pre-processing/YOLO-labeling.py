import csv
import os
import ast

# similarity = "0.7"
language = "english"
# language = "korean"
width_mul = 2.5
height_mul = 2.5
# CSV 파일 읽기
csv_file = f"/home/sooyong/datasets/original-datasets/image-csv/sim0.7/ocr_info_{language}.csv"
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # 이미지 파일명과 라벨 파일명
        dong_name = row['dong_name']
        category = row['category']
        image_filename = row['image_filename'].replace('.jpg', '')
        label_filename = f"{dong_name}_{category}_{image_filename}.txt"

        # 이미지의 width와 height 값 읽기
        image_width = float(row['width'])
        image_height = float(row['height'])
        
        # 좌표 정보 추출 (boxes에 여러 박스가 있을 수 있음)
        boxes = ast.literal_eval(row['boxes'])  # 문자열을 리스트로 변환

        # 라벨 저장 경로 설정
        label_dir = f'/home/sooyong/datasets/label-datasets/textbox({width_mul},{height_mul})/{language}_label'
        label_path = os.path.join(label_dir, label_filename)

        # 라벨 저장 경로 디렉토리가 없으면 생성
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            print(f"라벨 디렉토리 생성: {label_dir}")

        # 라벨 파일 열기 (쓰기 모드)
        with open(label_path, 'w') as label_file:
            # 여러 박스를 처리
            for box in boxes:
                # 좌표 정보 추출 (각 box는 4개의 좌표로 구성)
                x1, y1 = box[0]
                x2, y2 = box[1]
                x3, y3 = box[2]
                x4, y4 = box[3]
                
                # Bounding Box 중심 좌표와 너비/높이 계산
                x_center = (x1 + x2 + x3 + x4) / 4
                y_center = (y1 + y2 + y3 + y4) / 4
                width = abs(x2 - x1)
                height = abs(y3 - y2)
                
                # 정규화
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                width *= width_mul
                height *= height_mul

                width = 1 if width > 1 else width
                height = 1 if height > 1 else height
                
                # 클래스 ID (카테고리에서 추출, 예시로 'cafe'를 0번으로 설정)
                class_id = 0  # 필요에 따라 class_id 매핑
                
                # YOLO 형식 라벨 생성
                yolo_label = f"{class_id} {x_center} {y_center} {width} {height}\n"
                
                # 라벨 파일에 기록
                label_file.write(yolo_label)

        print(f"라벨 파일 저장 완료: {label_filename}")
