# OCR + LLM baselines for NavOCR

Code for the LLM rows in the [NavOCR](https://github.com/kc-ml2/NavOCR) paper.

NavOCR builds a training set for navigation-relevant text detection without
manual labelling. PP-OCRv5 reads the text in images of a place, and the
generator keeps the strings that match the place's name in open map resources,
such as OpenStreetMap. Compact detectors trained on this set then run on the
robot. A general-purpose LLM can also tell a shop name from an ad or a price, so
these experiments test whether an LLM could take over either step.

- **Table 1: can an LLM replace the map in the dataset generator?** The LLM
  generator gets the same OCR strings as the NavOCR generator, but no place
  name, and picks the place names itself. Both outputs are scored against text
  lists written by hand, so the gap shows what the map adds to label quality.
- **Table 3: can OCR plus an LLM replace the trained detector?** PP-OCRv5 finds
  every text box at inference time, and an LLM keeps the boxes whose text names
  a place. No detector is trained. AP against hand-drawn boxes and FPS are
  compared with the NavOCR-trained detectors. The target application, text
  mapping, needs about 10–15 FPS.

## Results

LLM rows show each model's best prompt.

### Table 1: dataset quality

Generator output against hand-written text lists, 304 images, micro average.

| Method | F1-score | Precision | Recall |
| --- | --- | --- | --- |
| NavOCR generator | 0.696 | 0.574 | 0.886 |
| Unfiltered OCR | 0.191 | 0.107 | 0.895 |
| LLM generator (Qwen3.5-9B, p3) | 0.420 | 0.289 | 0.772 |
| LLM generator (Qwen3.5-35B-A3B, p3) | 0.426 | 0.301 | 0.729 |

![Table 1 example](assets/table1_example.jpg)

### Table 3: detection

AP against hand-drawn boxes, 1,554 images.

| Method | AP@0.25 | AP@0.50 | FPS |
| --- | --- | --- | --- |
| PP-YOLOE (NavOCR) | 0.827 | 0.711 | 111.1 |
| RT-DETRv2 (NavOCR) | – | – | 142.9 |
| PP-OCRv5 (NavOCR) | 0.463 | 0.344 | 68.0 |
| PP-OCRv5 + LLM reranker (Qwen3.5-0.8B, p1) | 0.199 | 0.162 | 8.4 |
| PP-OCRv5 + LLM reranker (Qwen3.5-2B, p1) | 0.224 | 0.183 | 8.1 |
| PP-OCRv5 + LLM reranker (Qwen3.5-4B, p6) | 0.226 | 0.183 | 5.3 |
| PP-OCRv5 + LLM reranker (Qwen3.5-9B, p3) | 0.266 | 0.220 | 5.2 |

![Table 3 example](assets/navocr_vs_llm.jpg)

## Setup

PaddlePaddle (OCR) and PyTorch (LLM) need different CUDA builds (11.8 and
12.6), so they cannot share one environment. Create two conda environments
(python 3.10). The OCR steps run in `navocr-ocr` and write a JSONL file; the
LLM and scoring steps run in `navocr-llm` and read it.

OCR and LLM run on an NVIDIA GPU. Qwen3.5-9B fits on a 24 GB GPU;
Qwen3.5-35B-A3B needs an 80 GB GPU.

```bash
cd NeurIPS2026

conda create -n navocr-ocr python=3.10 -y
conda activate navocr-ocr
pip install -e ".[ocr]" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/

conda create -n navocr-llm python=3.10 -y
conda activate navocr-llm
pip install -e ".[llm]" --extra-index-url https://download.pytorch.org/whl/cu126

# PaddleDetection source is needed only for the fine-tuned PP-YOLOE variation.
mkdir -p third_party
git clone --branch release/2.8.1 --depth 1 \
    https://github.com/PaddlePaddle/PaddleDetection.git third_party/PaddleDetection
```

Set `PYTHONNOUSERSITE=1` in both so packages from `~/.local` do not leak in.
Versions are pinned in [`pyproject.toml`](pyproject.toml). Table 1 was run with
`torch==2.14.1`, Table 3 with `torch==2.13.0`.

Models need no separate download. Table 1's `llm-filter` and Table 3's
`filter` fetch the Qwen3.5 checkpoint from Hugging Face on their first run,
at the revision used in the paper
([`llm.py`](navocr_bench/llm.py)). PaddleOCR fetches its OCR models the same way.

The fine-tuned Table 3 detector artifacts are stored directly in this
repository under `weights/`:

- `weights/ppyoloe-navocr/model.pdparams`: PP-YOLOE training checkpoint used
  with the PaddleDetection source and standard PP-YOLOE-S config.
- `weights/ppocrv5-navocr/`: exported PP-OCRv5 mobile detector containing
  `inference.json`, `inference.pdiparams`, and `inference.yml`.

No other PaddleDetection training `output/` files or PP-OCRv5 training
checkpoints are required for inference.

`run_benchmark_detection.py` resolves `third_party/PaddleDetection` and the
paths under `weights/` relative to its own location, not the current working
directory or the user's home directory. Use `--paddledet-root`, `--config`,
`--weights`, or `--model-dir` only to override these defaults.

Test sets: [ML2/navocr-testsets](https://huggingface.co/datasets/ML2/navocr-testsets)
on Hugging Face (CC BY-NC 4.0, faces blurred).

```bash
hf download ML2/navocr-testsets --repo-type dataset --local-dir data
```

## Reproduce Table 1 (dataset quality)

Can an LLM replace the map in the dataset generator? Text-list F1 of the
generator output on 304 images.

**Data.** `data/text_selection_benchmark`: 304 images. `text_string_label.txt`
has one line per image:

```
image name, {text on the sign}, {place name from the map}, {translation}
```

All rows are scored against the second field. The NavOCR generator matches OCR
strings against the third and fourth fields (similarity ≥ 0.5). The LLM never
sees them.

**Run.** [`run_dataset_quality.py`](run_dataset_quality.py)

```bash
# 1. OCR
#    Reads every image with PP-OCRv5.
#    Output: out/dataset-quality/ocr_cache.jsonl (OCR strings per image)
conda activate navocr-ocr
python run_dataset_quality.py ocr --image-dir data/text_selection_benchmark/images \
    --gt-file data/text_selection_benchmark/text_string_label.txt --out-dir out/dataset-quality

# Steps 2-4 use the LLM environment.
conda activate navocr-llm

# 2a. LLM filter
#    The LLM picks the place names from each image's OCR strings.
#    One model load runs all six prompts.
#    Output: out/dataset-quality/sel.Qwen3.5-9B.p1.jsonl ... p6.jsonl
python run_dataset_quality.py llm-filter --out-dir out/dataset-quality \
    --model Qwen/Qwen3.5-9B --prompts p1,p2,p3,p4,p5,p6 --batch-size 8

# 2b. NavOCR filter
#    Matches the same OCR strings against the map place name and its translation.
#    Output: out/dataset-quality/sel.navocr.jsonl
python run_dataset_quality.py navocr-filter --out-dir out/dataset-quality \
    --gt-file data/text_selection_benchmark/text_string_label.txt \
    --place-threshold 0.5

# 3. Score
#    Compares every LLM and NavOCR selection with the hand-written labels.
#    Also scores the Unfiltered OCR row.
#    Output: out/dataset-quality/report.<model>.<prompt>.json (F1, precision, recall)
python run_dataset_quality.py score --out-dir out/dataset-quality \
    --gt-file data/text_selection_benchmark/text_string_label.txt

# 4. Table
#    Prints the Table 1 rows from the reports.
python run_dataset_quality.py table --out-dir out/dataset-quality
```

Steps 2a and 2b are independent filter variations over the same OCR cache; run
either one or both before scoring. For Qwen3.5-35B-A3B, repeat step 2a with
`--model Qwen/Qwen3.5-35B-A3B`, then steps 3 and 4.

## Reproduce Table 3 (detection)

Can PP-OCRv5 plus an LLM replace the trained detector? AP and FPS on 1,554
images.

**Data.** `data/detection_benchmark`: 1,554 images, 1,695 hand-drawn boxes in
`instances_test.json` (COCO, one category).

**Run.** [`run_benchmark_detection.py`](run_benchmark_detection.py)

```bash
# 1. Manifest
#    Fixes the image list and gives each image an id.
#    Output: out/detection/manifest.json
conda activate navocr-ocr
python run_benchmark_detection.py manifest --images data/detection_benchmark/images \
    --out out/detection/manifest.json

# 2a. OCR for the OCR + LLM baseline
#    PP-OCRv5 finds every text box and reads it.
#    Output: out/detection/ocr.jsonl (boxes, strings, confidences, OCR time per image)
python run_benchmark_detection.py ocr --manifest out/detection/manifest.json \
    --images data/detection_benchmark/images --out out/detection/ocr.jsonl

# 2b. Fine-tuned PP-YOLOE NavOCR variation
#    Uses the standard upstream config with the bundled fine-tuned checkpoint.
python run_benchmark_detection.py ppyoloe-detector \
    --manifest out/detection/manifest.json --images data/detection_benchmark/images \
    --out out/detection/ppyoloe-navocr.jsonl

# 2c. Fine-tuned PP-OCRv5 mobile_det NavOCR variation
python run_benchmark_detection.py ppocrv5-detector \
    --manifest out/detection/manifest.json --images data/detection_benchmark/images \
    --out out/detection/ppocrv5-navocr.jsonl

# Step 3 and the OCR + LLM FPS command use the LLM environment.
conda activate navocr-llm

# 3. LLM filter, once per model and prompt (optional)
#    The LLM keeps the strings that name a place.
#    Output: out/detection/Qwen3.5-9B.p3.jsonl (kept strings, LLM time per image)
python run_benchmark_detection.py filter --ocr out/detection/ocr.jsonl --model Qwen/Qwen3.5-9B \
    --prompt p3 --batch-size 8 --out out/detection/Qwen3.5-9B.p3.jsonl

# 4. AP
#    Scores any number of detector dumps or LLM filter outputs against the
#    hand-drawn boxes at IoU 0.25 and 0.50.
#    Output: out/detection/report.ap.json
python run_benchmark_detection.py ap --manifest out/detection/manifest.json \
    --gt data/detection_benchmark/instances_test.json \
    --ocr out/detection/ocr.jsonl \
    --pred out/detection/Qwen3.5-*.jsonl out/detection/ppyoloe-navocr.jsonl \
        out/detection/ppocrv5-navocr.jsonl \
    --out out/detection/report.ap.json

# 5a. OCR + LLM FPS
#    1 / (mean OCR time + mean LLM time).
#    Output: out/detection/fps.Qwen3.5-9B.p3.json
python run_benchmark_detection.py fps --manifest out/detection/manifest.json \
    --ocr out/detection/ocr.jsonl --filtered out/detection/Qwen3.5-9B.p3.jsonl \
    --out out/detection/fps.Qwen3.5-9B.p3.json

# 5b. Fine-tuned detector FPS
python run_benchmark_detection.py detector-fps --manifest out/detection/manifest.json \
    --pred out/detection/ppyoloe-navocr.jsonl \
    --out out/detection/fps.ppyoloe-navocr.json
python run_benchmark_detection.py detector-fps --manifest out/detection/manifest.json \
    --pred out/detection/ppocrv5-navocr.jsonl \
    --out out/detection/fps.ppocrv5-navocr.json
```

Steps 2a, 2b, and 2c are independent variations. Run only the variations to be
compared, then pass their existing output files to `ap`. Step 3 and `--ocr` in
Step 4 are needed only when an LLM filter output is included.

## Prompts

[`prompts.py`](navocr_bench/prompts.py) holds the six prompts (p1–p6).
`{items}` becomes the OCR strings as a JSON array.

## License

Code: [Apache License 2.0](../LICENSE), as in NavOCR. Data: CC BY-NC 4.0.
