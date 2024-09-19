import os
import csv

def count_csv_rows(csv_file):
    """CSV 파일의 헤더를 제외한 행(row)의 수를 계산"""
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더를 건너뜀
        row_count = sum(1 for row in reader)
    return row_count

def generate_summary_csv(category, input_root, output_csv_path):
    """폴더 내 모든 CSV 파일의 행 수를 계산하고 요약 CSV 파일 생성"""
    summary_data = []

    for lang in os.listdir(input_root):
        lang_input_path = os.path.join(input_root, lang)

        if not os.path.isdir(lang_input_path):
            continue

        for dong in os.listdir(lang_input_path):
            dong_input_path = os.path.join(lang_input_path, dong)
            
            if not os.path.isdir(dong_input_path):
                continue

            # 해당 카테고리 폴더 안의 모든 CSV 파일을 찾음
            for file in os.listdir(dong_input_path):
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(dong_input_path, file)
                    row_count = count_csv_rows(csv_file_path)
                    
                    # 요약 데이터 저장 (dong, dong, row_count)
                    summary_data.append([category, lang, dong, row_count])

    # 새로운 CSV 파일에 결과 저장
    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as summary_file:
        writer = csv.writer(summary_file)
        # writer.writerow(['category', 'language', 'dong', 'row_count'])  # 헤더 작성
        writer.writerows(summary_data)  # 각 폴더의 요약 데이터 작성

    print(f"요약 CSV 파일이 생성되었습니다: {output_csv_path}")

# 사용 예시
category = 'hospital'
input_root = f'/home/sooyong/datasets/OCR/results/{category}'  # INPUT_ROOT 경로 설정
output_csv_path = "/home/sooyong/datasets/OCR/count/summary.csv"  # 결과 CSV 파일 경로 설정
generate_summary_csv(category, input_root, output_csv_path)
