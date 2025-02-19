import argparse
import csv
import os
import logging
from paddleocr import PaddleOCR, draw_ocr
import paddle
import numpy as np
from PIL import Image
import utils

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,  # INFO에서 WARNING으로 변경
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)



def initialize_ocr(language='en', use_angle_cls=True, use_gpu=True):
    gpu_available = paddle.device.is_compiled_with_cuda()
    print("GPU available:", gpu_available)
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=language, use_gpu=use_gpu)


def perform_ocr(ocr, image_path):
    return ocr.ocr(image_path, cls=True)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"디렉토리 생성: {path}")
    else:
        logging.debug(f"디렉토리 존재: {path}")

def ensure_file(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w')
        logging.info(f"파일 생성: {file_path}")
    else:
        logging.debug(f"파일 존재: {file_path}")

def extract_words(result):
    return [(line[1][0].replace(" ", ""), line[1][1]) for item in result for line in item if line[1][0]]

def save_position_to_txt(x1, y1, x2, y2, file_path):
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            file.write(f'{x1[0]},{x1[1]},{x2[0]},{x2[1]},{y1[0]},{y1[1]},{y2[0]},{y2[1]}\n')
    except Exception as e:
        logging.error(f"TXT 저장 오류: {e}")

def save_mean_to_txt(mean, file_path):
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            file.write(f'{mean[0]},{mean[1]}\n')
    except Exception as e:
        logging.error(f"TXT 저장 오류: {e}")

def save_image_info_to_csv(file_path, image_filename, result, x1, y1, x2, y2, center_x, center_y, boxes, conf):
    try:
        file_exists = os.path.exists(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ['image_filename', 'result',
                          'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'boxes', 'conf']
                writer.writerow(header)
                logging.debug(f"헤더 작성 완료: {header}")
            writer.writerow([
                image_filename,
                result[0],
                x1, y1, x2, y2, center_x, center_y, boxes, conf
            ])
    except Exception as e:
        logging.error(f"CSV 저장 오류: {e}")


def make_file():
    ocr_dir = os.path.join(utils.OUTPUT_ROOT, "ocr")
    text_dir = os.path.join(utils.OUTPUT_ROOT, "text")
    
    ensure_directory(ocr_dir)
    ensure_directory(text_dir)
    
    # 이미 존재하는 파일 목록을 집합으로 가져오기
    existing_files = set(os.listdir(ocr_dir)) | set(os.listdir(text_dir))
    
    to_create = []
    input_files = [f for f in os.listdir(utils.INPUT_ROOT) if f.lower().endswith('.png')]
    
    for image_file in input_files:
        txt_file = image_file[:-4]  # '.png' 제거
        files = [
            f"ocr/{txt_file}_ocr_dete.txt",
            f"ocr/{txt_file}_ocr_mean.txt",
            f"text/{txt_file}_dete.txt",
            f"text/{txt_file}_mean.txt"
        ]
        
        for relative_path in files:
            if relative_path not in existing_files:
                full_path = os.path.join(utils.OUTPUT_ROOT, relative_path)
                to_create.append(full_path)
    
    for file_path in to_create:
        with open(file_path, 'w') as f:
            pass  # 빈 파일 생성

def ocr_first():
    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)

    csv_path = os.path.join(utils.OUTPUT_ROOT, f"ocr_info.csv")

    # 이미지 파일 처리
    for image_file in os.listdir(utils.INPUT_ROOT):
        txt_file = image_file.rstrip('.png')
        txt_path = os.path.join(utils.OUTPUT_ROOT, f"ocr/{txt_file}_ocr_dete.txt")
        mean_path = os.path.join(utils.OUTPUT_ROOT, f"ocr/{txt_file}_ocr_mean.txt")
        
        image_path = os.path.join(utils.INPUT_ROOT, image_file)

        try:
            result = perform_ocr(ocr, image_path)

            all_words = extract_words(result)
            
            boxes = [line[0] for item in result for line in item]
            scores = [line[1][1] for item in result for line in item]


            if len(all_words) == len(boxes):
                for i, single_word in enumerate(all_words):
                    x1, y1, x2, y2 = boxes[i]
                    center_x = (x1[0] + y1[0] + x2[0] + y2[0]) / 4
                    center_y = (x1[1] + y1[1] + x2[1] + y2[1]) / 4
                    save_position_to_txt(x1, y1, x2, y2, txt_path)
                    save_mean_to_txt(single_word, mean_path)
                    save_image_info_to_csv(csv_path, image_file, single_word, x1, y1, x2, y2, center_x, center_y, boxes[i], scores[i])


        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")

def compare(ocr_path, yolo_path):
    print('comparing start...')
    import ast  # Ensure ast is imported

    # 매칭된 OCR 결과를 저장할 딕셔너리
    matched_ocr = {}

    yolo_dict = {}
    with open(yolo_path, 'r', newline='', encoding='utf-8') as yolo_file:
        yolo_reader = csv.DictReader(yolo_file)
        for row in yolo_reader:
            yolo_file_name = row["image_filename"]
            conf = row["conf"]
            boxes = {
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
            }
            if yolo_file_name not in yolo_dict:
                yolo_dict[yolo_file_name] = []
            yolo_dict[yolo_file_name].append({'boxes': boxes, 'conf': conf})

    # 비교 결과를 저장할 디렉토리 설정
    COMPARE_OUTPUT_DIR = os.path.join(f"{utils.OUTPUT_ROOT}", "compare_results")
    ensure_directory(COMPARE_OUTPUT_DIR)


    with open(ocr_path, 'r', newline='', encoding='utf-8') as ocr_file:
        ocr_reader = csv.DictReader(ocr_file)
        for row in ocr_reader:
            ocr_file_name = row["image_filename"]
            ocr_result = row['result']
            try:
                # 좌표 문자열을 리스트로 변환
                ocr_x1 = ast.literal_eval(row['x1'])  # [211.0, 161.0]
                ocr_y1 = ast.literal_eval(row['y1'])  # [338.0, 155.0]
                ocr_x2 = ast.literal_eval(row['x2'])  # [339.0, 183.0]
                ocr_y2 = ast.literal_eval(row['y2'])  # [212.0, 189.0]
                ocr_center_x = float(row['center_x'])
                ocr_center_y = float(row['center_y'])
                ocr_conf = float(row['conf'])
                result = (ocr_result,ocr_conf)
                print(result)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Invalid coordinates or confidence for {ocr_file_name}, skipping.")
                continue

            if ocr_file_name in yolo_dict:
                for yolo_entry in yolo_dict[ocr_file_name]:
                    yolo_box = yolo_entry['boxes']
                    yolo_conf = yolo_entry['conf']
                    if (float(yolo_box['x1']) < float(ocr_center_x) < float(yolo_box['x2'])) and (float(yolo_box['y1']) < float(ocr_center_y) < float(yolo_box['y2'])) and float(ocr_conf) > 0.9:
                        # CSV 저장
                        # print(ocr_file_name)
                        txt_file = ocr_file_name.rstrip('.png')
                        txt_path = os.path.join(utils.OUTPUT_ROOT, f"text/{txt_file}_dete.txt")
                        mean_path = os.path.join(utils.OUTPUT_ROOT, f"text/{txt_file}_mean.txt")

                        ensure_file(txt_path)
                        ensure_file(mean_path)

                        save_mean_to_txt(result, mean_path)
                        save_position_to_txt(
                            ocr_x1,  # x1
                            ocr_y1,  # y1
                            ocr_x2,  # x2
                            ocr_y2,  # y2
                            txt_path
                        )

                        # 매칭된 OCR 결과 수집
                        if ocr_file_name not in matched_ocr:
                            matched_ocr[ocr_file_name] = {
                                'boxes': [],
                                'texts': [],
                                'ocr_conf': [],
                                'yolo_conf': []
                            }
                        # 박스 재구성: 네 개의 점으로 변환
                        reconstructed_box = [
                            [ocr_x1[0], ocr_x1[1]],
                            [ocr_y1[0], ocr_y1[1]],
                            [ocr_x2[0], ocr_x2[1]],
                            [ocr_y2[0], ocr_y2[1]]
                        ]
                        matched_ocr[ocr_file_name]['boxes'].append(reconstructed_box)
                        matched_ocr[ocr_file_name]['texts'].append(f"{ocr_result} (ocr_conf: {ocr_conf:.3f})")
                        matched_ocr[ocr_file_name]['ocr_conf'].append(ocr_conf)
                        matched_ocr[ocr_file_name]['yolo_conf'].append(float(yolo_conf))
                        break  # 만약 하나의 bounding box에만 저장하고 싶다면 break

    # 매칭된 OCR 결과를 사용하여 이미지 저장
    for image_file, ocr_data in matched_ocr.items():
        image_path = os.path.join(utils.INPUT_ROOT, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            boxes = ocr_data['boxes']
            texts = ocr_data['texts']
            ocr_conf = ocr_data['ocr_conf']
            yolo_conf = ocr_data['yolo_conf']
            # print(f"boxes: {boxes}")
            # print(f"texts: {texts}")
            # print(f"scores: {yolo_conf}")
            # 이미지에 OCR 결과 그리기
            im_show = draw_ocr(np.array(image), boxes, texts, yolo_conf, font_path=utils.FONT_PATH)
            im_show = Image.fromarray(im_show)

            # 저장 경로 설정
            output_image_filename = f"{os.path.splitext(image_file)[0]}_compare.jpg"
            output_image_path = os.path.join(COMPARE_OUTPUT_DIR, output_image_filename)
            im_show.save(output_image_path)
            logging.info(f"비교 이미지 저장 완료: {output_image_path}")
        except Exception as e:
            logging.error(f"비교 이미지 저장 오류 ({image_path}): {e}")


def main(output_root):
    # 실행 시 전달받은 OUTPUT_ROOT 값을 utils 모듈의 변수에 설정
    utils.OUTPUT_ROOT = output_root
    utils.INPUT_ROOT = f'{output_root}/images'

    # 이후 필요한 작업 수행
    print(f"OUTPUT_ROOT 설정 완료: {utils.OUTPUT_ROOT}")
    print(f"INPUT_ROOT 설정 완료: {utils.INPUT_ROOT}")

    make_file()
    ocr_first()
    compare(f"{utils.OUTPUT_ROOT}/ocr_info.csv", f"{utils.OUTPUT_ROOT}/yolo/yolo_info.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set OUTPUT_ROOT path for run.py")
    parser.add_argument(
        "output_root",
        type=str,
        help="Path to the OUTPUT_ROOT directory"
    )
    args = parser.parse_args()
    main(args.output_root)
