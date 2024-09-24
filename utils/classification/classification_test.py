import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import shutil  # 파일 이동 및 복사를 위한 라이브러리

# 사전 학습된 CLIP 모델과 프로세서 로드
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 분류할 클래스 라벨 정의
class_labels = ["a photo that are not contain signboard", "a photo of a store exterior"]

def classify_images_batch(image_paths, class_labels, model, processor, device, batch_size=16):
    """
    배치 단위로 이미지를 분류하는 함수
    """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"이미지 열기 실패: {path} | 오류: {e}")
                images.append(None)  # None을 추가하여 인덱스를 맞춤
        # 유효한 이미지만 처리
        valid_indices = [idx for idx, img in enumerate(images) if img is not None]
        valid_images = [img for img in images if img is not None]
        
        if not valid_images:
            continue  # 유효한 이미지가 없으면 다음 배치로 이동
        
        inputs = processor(text=class_labels, images=valid_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # [batch_size, num_labels]
            probs = logits_per_image.softmax(dim=1)      # 확률로 변환
        
        for idx, prob in zip(valid_indices, probs):
            store_exterior_prob = prob[1].item()  # 'a photo of a store exterior'의 확률
            results.append((batch_paths[idx], store_exterior_prob))
    return results
image_paths = [
    os.path.join("/home/sooyong/datasets/classification/test3", fname) for fname in os.listdir("/home/sooyong/datasets/classification/test3")
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
]
# batch_results = classify_images_batch(image_paths, class_labels, model, processor, device, batch_size=32)
# print(batch_results)

# 이미지가 저장된 디렉토리 경로
# # image_directory = "/home/sooyong/datasets/classification/high_similarity_images"
# image_directory = "/home/sooyong/datasets/OCR-results/sim0.5/Starfield/cafe/english/high_similarity_images"

# # 분류된 이미지를 저장할 두 개의 폴더 경로 정의
# no_sign_folder = os.path.join(image_directory, "no_sign")
# store_exterior_folder = os.path.join(image_directory, "store_exterior_images")

# # 폴더가 존재하지 않으면 생성
# os.makedirs(no_sign_folder, exist_ok=True)
# os.makedirs(store_exterior_folder, exist_ok=True)

# # 이미지 경로 수집


# # 배치 단위로 이미지 분류
batch_results = classify_images_batch(image_paths, class_labels, model, processor, device, batch_size=32)

# 분류 결과에 따라 이미지를 해당 폴더로 이동
for path, store_exterior_prob in batch_results:
    # 이미지 파일 이름 추출
    filename = os.path.basename(path)
    
    if store_exterior_prob >= 0.4:
        label = "a photo of a store exterior"
    else:
        label = "a photo that are not contain signboard"
    print(f"이미지: {filename} | 분류: {label} | 이동 완료 (확률: {store_exterior_prob:.4f})")
