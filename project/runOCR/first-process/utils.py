import os

# 폰트 파일 경로
FONT_PATH = '/home/sooyong/workspace/text_filtering_robot_mapping/shopOCR/fonts/AppleGothic.ttf'

# 유사도 임계값
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))

# 폴더, 카테고리, 언어를 환경 변수에서 가져오거나 기본값 사용
FOLDER = os.getenv('FOLDER', 'Starfield')
CATEGORY = os.getenv('CATEGORY', 'restaurant')
LANGUAGE_CODE = os.getenv('LANGUAGE', 'en')

LANGUAGE_FOLDER_MAP = {
    'en': 'english',
    'korean': 'korean'
}

LANGUAGE_FOLDER = LANGUAGE_FOLDER_MAP.get(LANGUAGE_CODE.lower(), 'english')

INPUT_ROOT = f'/home/sooyong/datasets/crawling-results/{FOLDER}/{CATEGORY}/{LANGUAGE_FOLDER}'
OUTPUT_ROOT = f'/home/sooyong/datasets/OCR-results/sim{SIMILARITY_THRESHOLD}/{FOLDER}/{CATEGORY}/{LANGUAGE_FOLDER}'
