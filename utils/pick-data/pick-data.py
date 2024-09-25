import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, deque

def create_dataset(
    source_base_dir,
    destination_base_dir,
    num_english,
    num_korean,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    # Define source paths
    english_label_dir = Path(source_base_dir) / "english_label"
    english_total_dir = Path(source_base_dir) / "english_total"
    korean_label_dir = Path(source_base_dir) / "korean_label"
    korean_total_dir = Path(source_base_dir) / "korean_total"

    # Function to extract store ID by removing the last '_' and number
    def extract_store_id(filename):
        # Split by '_' and remove the last part
        parts = filename.split('_')
        store_id = '_'.join(parts[:-1])
        return store_id

    # Function to group files by store
    def group_by_store(label_dir):
        store_dict = defaultdict(list)
        for f in label_dir.glob("*.txt"):
            store_id = extract_store_id(f.stem)
            store_dict[store_id].append(f.stem)
        return store_dict

    # 샘플 그룹화
    english_store_dict = group_by_store(english_label_dir)
    korean_store_dict = group_by_store(korean_label_dir)

    print(f"Total English stores available: {len(english_store_dict)}")
    print(f"Total Korean stores available: {len(korean_store_dict)}")

    # 전체 가게 리스트
    english_stores = list(english_store_dict.keys())
    korean_stores = list(korean_store_dict.keys())

    # 샘플 선택 함수
    def select_samples(store_dict, num_samples):
        selected = []
        store_ids = list(store_dict.keys())
        random.shuffle(store_ids)  # 가게 순서를 랜덤하게 섞기

        # Initialize deque for round-robin selection
        store_queue = deque(store_ids)

        # Keep track of selected samples per store to prevent over-selection
        selected_counts = defaultdict(int)

        while len(selected) < num_samples:
            if not store_queue:
                # 모든 가게를 순회했으나 여전히 샘플이 부족한 경우
                # 다시 모든 가게를 큐에 추가
                store_queue = deque(store_ids)
                # To prevent infinite loop in case of insufficient samples
                if all(selected_counts[store] >= len(store_dict[store]) for store in store_ids):
                    raise ValueError("Not enough unique samples to fulfill the request.")

            store = store_queue.popleft()
            available_samples = list(set(store_dict[store]) - set(selected))

            if available_samples:
                sample = random.choice(available_samples)
                selected.append(sample)
                selected_counts[store] += 1
            # If no available samples left in this store, skip adding it back to the queue

        return selected

    # 영어 샘플 선택
    if num_english <= len(english_stores):
        # 샘플 수가 가게 수보다 작거나 같을 경우, 각 가게에서 하나씩 선택
        selected_english_stores = random.sample(english_stores, num_english)
        selected_english = []
        for store in selected_english_stores:
            sample = random.choice(english_store_dict[store])
            selected_english.append(sample)
    else:
        # 샘플 수가 가게 수보다 많을 경우, 가게를 반복적으로 순회하며 샘플 선택
        selected_english = select_samples(english_store_dict, num_english)

    # 한국어 샘플 선택
    if num_korean <= len(korean_stores):
        # 샘플 수가 가게 수보다 작거나 같을 경우, 각 가게에서 하나씩 선택
        selected_korean_stores = random.sample(korean_stores, num_korean)
        selected_korean = []
        for store in selected_korean_stores:
            sample = random.choice(korean_store_dict[store])
            selected_korean.append(sample)
    else:
        # 샘플 수가 가게 수보다 많을 경우, 가게를 반복적으로 순회하며 샘플 선택
        selected_korean = select_samples(korean_store_dict, num_korean)

    # 결합 및 섞기
    combined_selected = selected_english + selected_korean
    random.shuffle(combined_selected)  # 섞기

    total = len(combined_selected)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = combined_selected[:train_end]
    val_files = combined_selected[train_end:val_end]
    test_files = combined_selected[val_end:]

    print(f"Total selected samples: {total}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Testing samples: {len(test_files)}")

    # Define destination subdirectories
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in splits.items():
        image_split_dir = Path(destination_base_dir) / "images" / split_name
        labels_split_dir = Path(destination_base_dir) / "labels" / split_name

        # Create directories if they don't exist
        image_split_dir.mkdir(parents=True, exist_ok=True)
        labels_split_dir.mkdir(parents=True, exist_ok=True)

        for file_stem in files:
            # Determine language to set source directories
            if file_stem in selected_english:
                label_src = english_label_dir / f"{file_stem}.txt"
                image_src = english_total_dir / f"{file_stem}.jpg"
            else:
                label_src = korean_label_dir / f"{file_stem}.txt"
                image_src = korean_total_dir / f"{file_stem}.jpg"

            # Define destination paths
            label_dst = labels_split_dir / f"{file_stem}.txt"
            image_dst = image_split_dir / f"{file_stem}.jpg"

            # Copy files
            shutil.copy2(label_src, label_dst)
            shutil.copy2(image_src, image_dst)

    print("Dataset creation complete.")

if __name__ == "__main__":

    box_size = "(2,2)"
    server = "epsilon"
    # Configuration
    SOURCE_BASE_DIR = f"/home/sooyong/datasets/original-datasets/textbox{box_size}"
    DESTINATION_BASE_DIR = f"/home/sooyong/datasets/yolo-dataset/textbox{box_size}"

    # Number of samples to select
    NUM_ENGLISH_SAMPLES = 11250 if server == "epsilon" else 7500 # 원하는 영어 샘플 수
    NUM_KOREAN_SAMPLES = 3750 if server == "epsilon" else 2500  # 원하는 한국어 샘플 수

    # Split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # Create the dataset
    create_dataset(
        source_base_dir=SOURCE_BASE_DIR,
        destination_base_dir=DESTINATION_BASE_DIR,
        num_english=NUM_ENGLISH_SAMPLES,
        num_korean=NUM_KOREAN_SAMPLES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42  # 재현성을 위해
    )
