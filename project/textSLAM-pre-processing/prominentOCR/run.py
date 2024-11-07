import csv
import os
import logging
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import utils

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def initialize_ocr(language='en', use_angle_cls=True, use_gpu=True):
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=language, use_gpu=use_gpu)


def perform_ocr(ocr, image_path):
    return ocr.ocr(image_path, cls=True)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"디렉토리 생성: {path}")
    else:
        logging.debug(f"디렉토리 존재: {path}")


def extract_words(result):
    return [(line[1][0].replace(" ", ""), line[1][1]) for item in result for line in item if line[1][0]]

def save_position_to_txt(x1, y1, x2, y2, file_path):
    try:
        file_exists = os.path.exists(file_path)
        if not file_exists:
            open(file_path, 'w')
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            file.write(f'{x1[0]},{x1[1]},{x2[0]},{x2[1]},{y1[0]},{y1[1]},{y2[0]},{y2[1]}\n')
    except Exception as e:
        logging.error(f"TXT 저장 오류: {e}")

def save_mean_to_txt(mean, conf, file_path):
    try:
        file_exists = os.path.exists(file_path)
        if not file_exists:
            open(file_path, 'w')
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            file.write(f'{mean},{conf}\n')
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
                result,
                x1, y1, x2, y2, center_x, center_y, boxes, conf
            ])
    except Exception as e:
        logging.error(f"CSV 저장 오류: {e}")


def save_prominent_sign_csv(file_path, image_filename, result, x1, y1, x2, y2, center_x, center_y, conf_ocr, conf_yolo):
    try:
        file_exists = os.path.exists(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ['image_filename', 'result',
                          'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'conf_ocr', 'conf_yolo']
                writer.writerow(header)
                logging.debug(f"헤더 작성 완료: {header}")
            writer.writerow([
                image_filename,
                result,
                x1, y1, x2, y2, center_x, center_y, conf_ocr, conf_yolo
            ])
    except Exception as e:
        logging.error(f"CSV 저장 오류: {e}")


def ocr_first():
    ensure_directory(utils.OUTPUT_ROOT)

    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)

    txt_path = os.path.join(utils.TXT_ROOT, f"ocr_info.csv")

    # 이미지 파일 처리
    for image_file in os.listdir(utils.INPUT_ROOT):
        if not image_file.lower().endswith(('.png')):
            continue
        
        txt_file = image_file.rstrip('.png')
        txt_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_dete.txt")
        mean_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_mean.txt")
        
        image_path = os.path.join(utils.INPUT_ROOT, image_file)
        output_image_filename = f"{os.path.splitext(image_file)[0]}_result.jpg"
        output_image_path = os.path.join(
            utils.OUTPUT_ROOT, output_image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
            result = perform_ocr(ocr, image_path)

            all_words = extract_words(result)
            
            boxes = [line[0] for item in result for line in item]
            save_mean_to_txt(all_words, mean_path)
            # 이미지 저장

            im_show = draw_ocr(np.array(image), boxes, all_words[0],
                               all_words[1], font_path=utils.FONT_PATH)
            im_show = Image.fromarray(im_show)

            if len(all_words) == len(boxes):
                for i, single_word in enumerate(all_words):
                    x1 = boxes[i][0]
                    y1 = boxes[i][1]
                    x2 = boxes[i][2]
                    y2 = boxes[i][3]
                    save_position_to_txt(x1, y1, x2, y2, txt_path)


        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")

def compare(save_path, ocr_path, yolo_path):
    print('comparing start...')
    import ast  # Ensure ast is imported

    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)

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
    COMPARE_OUTPUT_DIR = os.path.join("/home/sooyong/datasets/textSlam/prominent/", "compare_results")
    ensure_directory(COMPARE_OUTPUT_DIR)

    # 매칭된 OCR 결과를 저장할 딕셔너리
    matched_ocr = {}

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
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Invalid coordinates or confidence for {ocr_file_name}, skipping.")
                continue

            if ocr_file_name in yolo_dict:
                for yolo_entry in yolo_dict[ocr_file_name]:
                    yolo_box = yolo_entry['boxes']
                    yolo_conf = yolo_entry['conf']
                    if (float(yolo_box['x1']) < float(ocr_center_x) < float(yolo_box['x2'])) and (float(yolo_box['y1']) < float(ocr_center_y) < float(yolo_box['y2'])):
                        # CSV 저장
                        print(ocr_file_name)
                        txt_file = ocr_file_name.rstrip('.png')
                        txt_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_dete.txt")
                        mean_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_mean.txt")
                        save_mean_to_txt(ocr_result, ocr_conf, mean_path)

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


def main():
    # ocr_first()
    compare("/home/sooyong/datasets/textSlam/prominent/txt", "/home/sooyong/datasets/output-3/ocr_info.csv", "/home/sooyong/datasets/output-3/yolo_info.csv")


if __name__ == "__main__":
    main()
