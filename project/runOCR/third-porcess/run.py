import csv
import os
import logging
import re
import shutil
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import utils
import textdistance as td

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def initialize_ocr(language='en', use_angle_cls=True):
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=language)


def perform_ocr(ocr, image_path):
    return ocr.ocr(image_path, cls=True)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"디렉토리 생성: {path}")
    else:
        logging.debug(f"디렉토리 존재: {path}")


def extract_words(result):
    return [line[1][0].replace(" ", "").lower() for item in result for line in item if line[1][0]]


def save_image_info_to_csv(file_path, image_filename, result, x1, y1, x2, y2, center_x, center_y, conf):
    try:
        file_exists = os.path.exists(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ['image_filename', 'result',
                          'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'confidence']
                writer.writerow(header)
                logging.debug(f"헤더 작성 완료: {header}")
            writer.writerow([
                image_filename,
                result,
                x1, y1, x2, y2, center_x, center_y, conf
            ])
    except Exception as e:
        logging.error(f"CSV 저장 오류: {e}")


def main():
    ensure_directory(utils.OUTPUT_ROOT)

    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)

    csv_path = os.path.join(utils.CSV_ROOT, f"ocr_info.csv")

    # 이미지 파일 처리
    for image_file in os.listdir(utils.INPUT_ROOT):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(utils.INPUT_ROOT, image_file)
        output_image_filename = f"{os.path.splitext(image_file)[0]}_result.jpg"
        output_image_path = os.path.join(
            utils.OUTPUT_ROOT, output_image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
            result = perform_ocr(ocr, image_path)

            all_words = extract_words(result)
            # if not all_words:
            #     logging.info(f"유사도가 높은 OCR 단어가 없어 이미지 저장하지 않음: {image_path}")
            #     continue

            boxes = [line[0] for item in result for line in item]
            scores = [line[1][1] for item in result for line in item]

            # 이미지 저장
            im_show = draw_ocr(np.array(image), boxes, all_words,
                               scores, font_path=utils.FONT_PATH)
            im_show = Image.fromarray(im_show)
            im_show.save(output_image_path)
            logging.info(f"이미지 저장 완료: {output_image_path}")

            # CSV 저장
            # print(all_words)
            # print(boxes[0][0])
            for i, single_word in enumerate(all_words):
                x1 = boxes[i][0]
                y1 = boxes[i][1]
                x2 = boxes[i][2]
                y2 = boxes[i][3]
                center_x = (x1[0] + y1[0] + x2[0] + y2[0]) / 4
                center_y = (x1[1] + y1[1] + x2[1] + y2[1]) / 4
                # print(single_word, x1, y1, x2, y2, center_x, center_y)
                save_image_info_to_csv(csv_path, image_file, single_word,
                                       x1, y1, x2, y2, center_x, center_y,  scores[i])

        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")


if __name__ == "__main__":
    main()
