import os
import shutil
import random
from pathlib import Path

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

    # Collect filenames (without extensions)
    english_files = [f.stem for f in english_label_dir.glob("*.txt")]
    korean_files = [f.stem for f in korean_label_dir.glob("*.txt")]

    print(f"Total English samples available: {len(english_files)}")
    print(f"Total Korean samples available: {len(korean_files)}")

    # Ensure enough samples are available
    if num_english > len(english_files):
        raise ValueError(f"Requested {num_english} English samples, but only {len(english_files)} available.")
    if num_korean > len(korean_files):
        raise ValueError(f"Requested {num_korean} Korean samples, but only {len(korean_files)} available.")

    # Randomly select samples
    selected_english = random.sample(english_files, num_english)
    selected_korean = random.sample(korean_files, num_korean)

    # Combine selections
    combined_selected = selected_english + selected_korean
    random.shuffle(combined_selected)  # Shuffle combined list

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
    # Configuration
    SOURCE_BASE_DIR = "/home/sooyong/datasets/original-datasets/textbox(2,2)/sim0.7"
    DESTINATION_BASE_DIR = "/home/sooyong/datasets/yolo-dataset/textbox(2,2)/sim0.7_training_10k(7.5|2.5;6:2:2)"

    # Number of samples to select
    NUM_ENGLISH_SAMPLES = 7500 # Set your desired number
    NUM_KOREAN_SAMPLES = 2500 # Set your desired number

    # Split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    # Create the dataset
    create_dataset(
        source_base_dir=SOURCE_BASE_DIR,
        destination_base_dir=DESTINATION_BASE_DIR,
        num_english=NUM_ENGLISH_SAMPLES,
        num_korean=NUM_KOREAN_SAMPLES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42  # For reproducibility
    )
