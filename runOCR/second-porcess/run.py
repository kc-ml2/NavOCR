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
    return [line[1][0].replace(" ", "").lower() for item in result for line in item]

def save_image_info_to_csv(csv_file_path, dong_name, category, lang, image_filename, result, ocr_result, similarity, width, height, center, boxes):
    try:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ['lang', 'dong_name', 'category', 'image_filename', 'result', 'ocr_result', 'similarity', 'width', 'height', 'center', 'boxes']
                writer.writerow(header)
                logging.debug(f"헤더 작성 완료: {header}")
            writer.writerow([
                lang, 
                dong_name, 
                category, 
                image_filename, 
                result,
                ocr_result, 
                similarity,
                width,  # 이미지 너비 추가
                height,  # 이미지 높이 추가
                center,
                boxes
            ])
        logging.debug(f"CSV 저장 완료: {csv_file_path}")
    except Exception as e:
        logging.error(f"CSV 저장 오류 ({csv_file_path}): {e}")

def parse_filename(filename):
    match = re.match(r'([^_]+)_([^_]+)_([^_]+)_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        dong_name, category, store_name, image_id, ext = match.groups()
        image_filename = f"{store_name}_{image_id}.{ext}"  # 수정된 부분
        return dong_name, category, store_name, image_filename
    else:
        logging.warning(f"파일 이름 형식이 올바르지 않습니다: {filename}")
        return None, None, None, None

def main():
    # 출력 디렉토리 설정
    ensure_directory(utils.OUTPUT_ROOT)
    
    # OCR 초기화
    ocr = initialize_ocr(language=utils.LANGUAGE_CODE, use_angle_cls=True)
    
    # CSV 파일 경로
    csv_path = os.path.join(utils.CSV_ROOT, f"ocr_info_{utils.LANGUAGE_FOLDER}.csv")
    
    # 이미지 파일 처리
    for image_file in os.listdir(utils.INPUT_ROOT):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        dong_name, category, store_name, image_filename = parse_filename(image_file)
        if not all([dong_name, category, store_name, image_filename]):
            continue
        
        image_path = os.path.join(utils.INPUT_ROOT, image_file)
        output_image_filename = f"{os.path.splitext(image_file)[0]}_result.jpg"
        output_image_path = os.path.join(utils.OUTPUT_ROOT, output_image_filename)
        
        logging.info(f"처리 중: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size  # 이미지 크기 추출
            result = perform_ocr(ocr, image_path)
            
            # store_name 전처리
            store_clean = re.sub(r'\b\w*(Starfield|스타필드|coex)\w*\b', '', store_name, flags=re.IGNORECASE)
            stores = re.sub(' +', ' ', store_clean).strip()
            store_words = stores.split()
            store_combined = store_clean.replace(" ", "").lower()
            all_words = extract_words(result)
            
            similarity_threshold = utils.SIMILARITY_THRESHOLD
            boxes = [line[0] for item in result for line in item]
            scores = [line[1][1] for item in result for line in item]
            
            filtered_words = []
            filtered_boxes = []
            filtered_scores = []
            filtered_similarities = []
            box_centers = []
            
            for i, single_word in enumerate(all_words):
                if not single_word:
                    continue
                
                logging.debug(f"OCR로 감지된 단어: {single_word}")
                max_similarity = 0
                best_match_word = None
                
                for word in store_words:
                    similarity = td.levenshtein.normalized_similarity(word.lower(), single_word.lower())
                    logging.debug(f"'{word}'와 유사도: {similarity}")
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_word = word
                
                similarity_combined = td.levenshtein.normalized_similarity(store_combined, single_word.lower())
                logging.debug(f"'{store_combined}'와 유사도: {similarity_combined}")
                
                if similarity_combined > max_similarity:
                    max_similarity = similarity_combined
                    best_match_word = store_combined
                
                if max_similarity >= similarity_threshold:
                    filtered_words.append(f'{single_word}({best_match_word})')
                    filtered_boxes.append(boxes[i])
                    filtered_scores.append(scores[i])
                    filtered_similarities.append(max_similarity)
                    
                    # 박스의 중심점 계산
                    box = np.array(boxes[i])
                    center = np.mean(box, axis=0).tolist()
                    box_centers.append(center)
                    logging.debug(f'center: {center}')
            
            if filtered_boxes and filtered_words:
                im_show = draw_ocr(
                    np.array(image), 
                    filtered_boxes, 
                    filtered_words, 
                    filtered_scores, 
                    font_path=utils.FONT_PATH
                )
                im_show = Image.fromarray(im_show)
                im_show.save(output_image_path)
                logging.info(f"이미지 저장 완료: {output_image_path}")
                
                # CSV 저장
                save_image_info_to_csv(
                    csv_path, 
                    dong_name, 
                    category, 
                    utils.LANGUAGE_CODE, 
                    image_filename,  # 수정된 image_filename 사용
                    filtered_words, 
                    filtered_scores, 
                    filtered_similarities, 
                    width,  # 이미지 너비 전달
                    height,  # 이미지 높이 전달
                    box_centers, 
                    filtered_boxes
                )
            else:
                logging.info(f"유사도가 높은 OCR 단어가 없어 이미지 저장하지 않음: {image_path}")
        
        except Exception as e:
            logging.error(f"이미지 처리 오류 ({image_path}): {e}")

if __name__ == "__main__":
    main()
