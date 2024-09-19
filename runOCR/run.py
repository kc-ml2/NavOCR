import csv
import os
import logging
import re
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import textdistance as td
import shutil  # 파일 복사를 위해 추가
import utils

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def initialize_ocr(language='en', use_angle_cls=True):
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=language)
    return ocr

def perform_ocr(ocr, image_path):
    result = ocr.ocr(image_path, cls=True)
    return result

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # logging.info(f"디렉토리 생성: {path}")
    else:
        logging.debug(f"디렉토리 존재: {path}")

def extract_words(result):
    word_result = []
    # 단어 및 좌표 처리
    for item in result:
        for line in item:
            word = line[1][0].replace(" ", "").lower()
            word_result.append(word)
    return word_result

def save_image_info_to_csv(csv_file_path, dong_name, category, lang, image_filename, result, ocr_result, similarity, center, boxes):
    try:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Serialize 'result' and 'center' as JSON strings
            if not file_exists:
                header = ['lang', 'dong_name', 'category', 'image_filename', 'result', 'ocr_result', 'similarity', 'center', 'boxes']
                writer.writerow(header)
                # logging.debug(f"헤더 작성 완료: {header}")
            writer.writerow([
                lang, 
                dong_name, 
                category, 
                image_filename, 
                result,
                ocr_result, 
                similarity,
                center,
                boxes
            ])
        # logging.debug(f"CSV 저장 완료: {csv_file_path}")
    except Exception as e:
        logging.error(f"CSV 저장 오류 ({csv_file_path}): {e}")

ocr = initialize_ocr(language=utils.LANGUAGE, use_angle_cls=True)

# 유사도가 높은 이미지들을 복사할 디렉토리 설정
high_similarity_dir = os.path.join(utils.OUTPUT_ROOT, 'high_similarity_images')
ensure_directory(high_similarity_dir)  # 디렉토리 생성

for dong in os.listdir(utils.INPUT_ROOT):
    # if dong != "코엑스":
    #     continue
    dong_input_path = os.path.join(utils.INPUT_ROOT, dong)
    dong_output_path = os.path.join(utils.OUTPUT_ROOT, dong)
    ensure_directory(dong_output_path)

    if not os.path.isdir(dong_input_path):
        # logging.warning(f"디렉토리가 존재하지 않습니다: {dong_input_path}")
        continue
    for shop in os.listdir(dong_input_path):
        shop_input_path = os.path.join(dong_input_path, shop)
        shop_output_path = os.path.join(dong_output_path, shop)
        ensure_directory(shop_output_path)
        
        if not os.path.isdir(shop_input_path):
            # logging.warning(f"디렉토리가 존재하지 않습니다: {shop_input_path}")
            continue

        store_name = shop  
        
        # CSV 파일 경로 설정
        csv_path = os.path.join(dong_output_path, 'ocr_info.csv')
        for image_file in os.listdir(shop_input_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(shop_input_path, image_file)
                output_image_filename = f"{os.path.splitext(image_file)[0]}_result.jpg"
                output_image_path = os.path.join(shop_output_path, output_image_filename)
                
                # logging.info(f"처리 중: {image_path}")
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    result = perform_ocr(ocr, image_path)

                    store = re.sub(r'\b\w*(Starfield|스타필드|coex)\w*\b', '', store_name, flags=re.IGNORECASE)
                    stores = re.sub(' +', ' ', store).strip()
                    store_words = stores.split()
                    
                    store_combined = store.replace(" ", "").lower()

                    all_words = extract_words(result)

                    similarity_mat = np.zeros([len(all_words), len(store_words) + 1])  # 가게 이름 단어 수 + 전체 결합 단어

                    # 유사도를 체크할 임계값 설정
                    similarity_threshold = utils.SIMILARITY_THRESHOLD

                    # OCR로 얻은 박스 좌표 추출
                    boxes = [line[0] for item in result for line in item]
                    scores = [line[1][1] for item in result for line in item]
                    
                        
                    filtered_words = []
                    filtered_boxes = []
                    filtered_scores = []
                    filtered_similarities = []

                    for i, single_word in enumerate(all_words):
                        if single_word:
                            # logging.debug(f"OCR로 감지된 단어: {single_word}")

                            max_similarity = 0
                            best_match_word = None  # 최적의 매칭 단어를 저장할 변수 초기화

                            for j, word in enumerate(store_words):
                                similarity = td.levenshtein.normalized_similarity(word.lower(), single_word.lower())
                                similarity_mat[i][j] = similarity
                                # logging.debug(f"'{word}'와 유사도: {similarity}")
                                
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_match_word = word  # 현재 단어를 최적의 매칭 단어로 저장

                            # 결합된 형태와의 유사도 계산
                            similarity_combined = td.levenshtein.normalized_similarity(store_combined.lower(), single_word.lower())
                            similarity_mat[i][len(store_words)] = similarity_combined
                            # logging.debug(f"'{store_combined}'와 유사도: {similarity_combined}")
                            
                            # 결합된 형태와의 유사도와 개별 단어 유사도 중에서 가장 높은 값을 선택
                            if similarity_combined > max_similarity:
                                max_similarity = similarity_combined
                                best_match_word = store_combined  # 결합된 형태를 최적의 매칭 단어로 저장

                            # 임계값 이상이면 필터링된 결과에 추가
                            if max_similarity >= similarity_threshold:
                                filtered_words.append(f'{single_word}({best_match_word})')
                                filtered_boxes.append(boxes[i])  # 해당 단어의 박스를 필터링된 결과에 추가
                                filtered_scores.append(scores[i])  # 해당 단어의 점수를 필터링된 결과에 추가
                                filtered_similarities.append(max_similarity)


                    # 모든 단어를 처리한 후 필터링된 단어가 있는 경우에만 이미지 저장
                    if filtered_boxes and filtered_words:
                        im_show = draw_ocr(
                            image, 
                            filtered_boxes, 
                            filtered_words, 
                            filtered_scores, 
                            font_path=utils.FONT_PATH
                        )
                        box_center = []
                        for box in filtered_boxes:
                            box = np.array(box)
                            center = np.mean(box, axis=0)  # 중심점 계산
                            box_center.append(center.tolist())  # Convert to list
                            # logging.debug(f'center: {center.tolist()}')  # Log as list
                        
                        im_show = Image.fromarray(im_show)
                        im_show.save(output_image_path)
                        # logging.info(f"이미지 저장 완료: {output_image_path}")
                        
                        # CSV 저장
                        save_image_info_to_csv(csv_path, dong, shop, utils.LANGUAGE, image_file, filtered_words, filtered_scores, filtered_similarities, box_center, filtered_boxes)
                        
                        # 유사도가 높은 이미지의 원본 파일을 별도 디렉토리에 복사
                        destination_path = os.path.join(high_similarity_dir, f"{dong}_{shop}_{image_file}")
                        shutil.copy2(image_path, destination_path)
                        logging.info(f"유사도가 높은 이미지 복사 완료: {destination_path}")
                    else:
                        logging.info(f"유사도가 높은 OCR 단어가 없어 이미지 저장하지 않음: {image_path}")

                except Exception as e:
                    logging.error(f"이미지 처리 오류 ({image_path}): {e}")
