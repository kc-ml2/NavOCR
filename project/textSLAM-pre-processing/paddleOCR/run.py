import csv
import os
import logging
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import utils
import time

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
                if mean[1] > 0.9:
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
    COMPARE_OUTPUT_DIR = os.path.join(utils.OUTPUT_ROOT, "compare_results")
    TEXT_OUTPUT_DIR = os.path.join(utils.OUTPUT_ROOT, "text")

    ensure_directory(TEXT_OUTPUT_DIR)

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
            scores = [line[1][1] for item in result for line in item]
            ensure_file(txt_path)
            ensure_file(mean_path)
            save_mean_to_txt(all_words, mean_path)

            filtered_words = []
            filtered_boxes = []
            filtered_scores = []

            ensure_directory(COMPARE_OUTPUT_DIR)


            for i, single_word in enumerate(all_words):
                filtered_words.append(f'{single_word[0]}')
                filtered_boxes.append(boxes[i]) 
                filtered_scores.append(scores[i])
                # print(f"filtered_words: {filtered_words}")
                # print(f"filtered_boxes: {filtered_boxes}")
                # print(f"filtered_scores: {filtered_scores}")
                image = Image.open(image_path).convert('RGB')
                if filtered_boxes and filtered_words:
                    im_show = draw_ocr(
                        image, 
                        filtered_boxes, 
                        filtered_words, 
                        filtered_scores, 
                        font_path=utils.FONT_PATH
                    )
                # # 이미지에 OCR 결과 그리기
                # im_show = draw_ocr(np.array(image), filtered_boxes, filtered_words, filtered_scores, font_path=utils.FONT_PATH)
                    im_show = Image.fromarray(im_show)

                # 저장 경로 설정
                output_image_filename = f"{os.path.splitext(image_file)[0]}_compare.jpg"
                output_image_path = os.path.join(COMPARE_OUTPUT_DIR, output_image_filename)
                im_show.save(output_image_path)

                word, conf = single_word
                if conf > 0.9:
                    x1 = boxes[i][0]
                    y1 = boxes[i][1]
                    x2 = boxes[i][2]
                    y2 = boxes[i][3]
                    save_position_to_txt(x1, y1, x2, y2, txt_path)

        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")

def main():
    start_time = time.time()  # 시작 시간 기록
    ocr_first()
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 총 걸린 시간 계산
    print(f"총 실행 시간: {elapsed_time:.2f}초")

if __name__ == "__main__":
    main()
