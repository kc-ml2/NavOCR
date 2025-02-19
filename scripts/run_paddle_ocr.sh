#!/bin/bash

# 기본 경로 설정
BASE_OUTPUT_ROOT="/mnt/sda/coex_data/track_fail"
SCRIPT_PATH="/home/sooyong/workspace/shopOCR/project/textSLAM-pre-processing/paddleOCR/run.py"  # Python 스크립트 파일 경로
FONT_PATH="/home/sooyong/workspace/shopOCR/fonts/AppleGothic.ttf"  # 폰트 경로

# 변경 가능한 앞부분 단어 리스트
PREFIXES=("columbia" "converse" "lifefourcut", "plantude", "shake")

# 반복 실행
for prefix in "${PREFIXES[@]}"; do
    for i in {1..3}; do
        # 디렉토리 이름 변경
        OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${prefix}_${i}"
        INPUT_ROOT="${OUTPUT_ROOT}/images"
        TXT_ROOT="${OUTPUT_ROOT}/text"

        # 필요한 디렉토리가 없으면 생성
        mkdir -p "$INPUT_ROOT" "$TXT_ROOT"

        # Python 스크립트 실행
        python "$SCRIPT_PATH" \
            --font_path "$FONT_PATH" \
            --language_code "en" \
            --output_root "$OUTPUT_ROOT" \
            --input_root "$INPUT_ROOT" \
            --txt_root "$TXT_ROOT"
    done
done
