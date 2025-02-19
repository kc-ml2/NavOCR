# Prominent Text OCR

### Project
##### runOCR
- first-process: save OCR results hierarchically and create scv file
- **second-process**: accumulate OCR results and create csv file
- third-process: compare OCR results with YOLO results

##### YOLO-fine-tune
- fine-tuning code for YOLO model

##### runYOLO
- Runs the fine-tuned YOLO model for prominent sign

##### textSLAM-pre-processing
- paddleOCR: OCR every texts
- prominentOCR: OCR only prominent texts

### utils
##### classification
- use CLIP model to classify only outdoor images of store

## Create environment

#### create new environment for paddle OCR (python 3.7.16)

download required settings

```
pip install -r requirements_ocr.txt
```

run OCR code

```
./scripts/run_first_ocr.sh
./scripts/run_second_ocr.sh
```

#### create new environment for YOLO

download required settings

```
pip install -r requirements_yolo.txt
```

#### create new environment for CLIP

download required settings

```
pip install -r requirements_clip.txt
```

