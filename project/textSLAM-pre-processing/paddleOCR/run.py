import csv
import os
import logging
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import utils

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
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            file.write(f'{x1[0]},{x1[1]},{x2[0]},{x2[1]},{y1[0]},{y1[1]},{y2[0]},{y2[1]}\n')
    except Exception as e:
        logging.error(f"TXT 저장 오류: {e}")

def save_mean_to_txt(means, file_path):
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            for mean in means:
                file.write(f'{mean[0]},{mean[1]}\n')
    except Exception as e:
        logging.error(f"TXT 저장 오류: {e}")

def ensure_file(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w')
    else:
        logging.debug(f"파일 존재: {file_path}")



def ocr_first():
    ensure_directory(utils.OUTPUT_ROOT)

    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)


    # 이미지 파일 처리
    for image_file in os.listdir(utils.INPUT_ROOT):
        if not image_file.lower().endswith(('.png')):
            continue
        txt_file = image_file.rstrip('.png')
        txt_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_dete.txt")
        mean_path = os.path.join(utils.TXT_ROOT, f"{txt_file}_mean.txt")
        image_path = os.path.join(utils.INPUT_ROOT, image_file)
        try:
            result = perform_ocr(ocr, image_path)
            
            all_words = extract_words(result)
            
            boxes = [line[0] for item in result for line in item]
            ensure_file(txt_path)
            ensure_file(mean_path)
            save_mean_to_txt(all_words, mean_path)

            for i, single_word in enumerate(all_words):
                x1 = boxes[i][0]
                y1 = boxes[i][1]
                x2 = boxes[i][2]
                y2 = boxes[i][3]
                save_position_to_txt(x1, y1, x2, y2, txt_path)

        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")

def main():
    ocr_first()

if __name__ == "__main__":
    main()
