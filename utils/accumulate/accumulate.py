

import itertools
import os
import shutil

similarities = ["sim0.5", "sim0.6", "sim0.7"]
places = ["Starfield", "general"]
categories = ["restaurant", "cafe", "bakery", "hospital", "cloth"]
# categories = ["cloth"]
languages = ["english", "korean"]

# 모든 조합 생성
combinations = itertools.product(similarities, places, categories, languages)

for similarity, place, category, language in combinations:
    save_directory = f"/home/sooyong/datasets/OCR-results/high_similarity_images/{similarity}/{language}/"

    # 디렉토리 존재 여부 확인
    if not os.path.exists(save_directory):
        print(f"디렉토리가 존재하지 않습니다: {save_directory}")
        continue  # 다음 조합으로 넘어감

    accumulate_directory = os.path.join(save_directory, f"{place}_{category}_datasets")
    print(f"accumulate_directory: {accumulate_directory}")
    destination = os.path.join(save_directory, f"{language}_total")
    print(f"destination: {destination}")

    if not os.path.exists(destination):
        os.makedirs(destination)
        print(f"대상 폴더 생성: {destination}")

    if os.path.exists(accumulate_directory) and os.path.isdir(accumulate_directory):
        for filename in os.listdir(accumulate_directory):
            file_path = os.path.join(accumulate_directory, filename)
            if os.path.isfile(file_path):
                destination_file_path = os.path.join(destination, filename)
                if not os.path.exists(destination_file_path):
                    try:
                        shutil.copy(file_path, destination_file_path)
                        print(f"파일 복사 완료: {filename}")
                    except Exception as e:
                        print(f"파일 복사 실패: {filename} | 오류: {e}")
                else:
                    print(f"파일 {filename}은(는) 이미 존재하여 복사하지 않음")
    else:
        print(f"경로가 존재하지 않거나 디렉토리가 아님: {accumulate_directory}")