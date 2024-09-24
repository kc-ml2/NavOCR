#!/bin/bash

# 파라미터 설정
FOLDERS=("Starfield" "general")
CATEGORIES=("restaurant" "cafe" "cloth" "hospital" "bakery")
LANGUAGES=("en" "korean")
SIMILARITY_THRESHOLD=("0.6")

# 모든 조합에 대해 run.py 실행
# for folder in "${FOLDERS[@]}"; do
#   for category in "${CATEGORIES[@]}"; do
#     for language in "${LANGUAGES[@]}"; do
#       echo "실행 중: 폴더=${folder}, 카테고리=${category}, 언어=${language}"
      
#       # 환경 변수 설정 후 run.py 실행
#       FOLDER=$folder CATEGORY=$category LANGUAGE=$language python3 /home/sooyong/datasets/shopOCR/runOCR/run.py
      
#       # 실행 완료 메시지
#       echo "완료: 폴더=${folder}, 카테고리=${category}, 언어=${language}"
#     done
#   done
# done

for similarity in "${SIMILARITY_THRESHOLD[@]}"; do
  for language in "${LANGUAGES[@]}"; do
    echo "실행 중: 유사도=${similarity}, 언어=${language}"
    
    # 환경 변수 설정 후 run.py 실행
    SIMILARITY_THRESHOLD=$similarity LANGUAGE=$language python3 /home/sooyong/datasets/shopOCR/runOCR/run.py
    
    # 실행 완료 메시지
    echo "완료: 유사도=${similarity}, 언어=${language}"
  done
done