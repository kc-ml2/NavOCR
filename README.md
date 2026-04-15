# NavOCR

## Text Detection for Navigation!

NavOCR is an open-source project that provides a lightweight text detection model for navigation.  
Other publicly available OCR models often work too well and detect texts unrelated to navigation, such as advertisements, logos, or price tags.
NavOCR detects only the text that is necessary for navigation, such as signboards, directional guides, and room numbers.  
We provide the full pipeline for model training (including data crawling, dataset preprocessing, and fine-tuning).

❗This repository is currently under heavy refactoring and development.
Please note that it may contain unstable components.
Improvements and updates will be released soon.

<p align="center">
    <img src="./example.svg" alt="NavOCR_example"
</p>

<p align="center">
    <img src="./NavOCR.gif" alt="NavOCR" width="572"/>
</p>


## How to Use

### Download Model
Our model is included in this repo. So, clone this repo to download the model!  
The current model supports detection of store signboards only.
Detection of other navigation-relevant text types will be supported in future updates.

```bash
git clone git@github.com:kc-ml2/NavOCR.git
```

### Prerequisite

Install PaddleDetection following [offical guide](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/INSTALL.md).

Install PaddleOCR:
```bash
pip install paddleocr==2.8.1
```

### Download Testset
Not required if you use ROS node

```bash
# Setup python env
pip install gdown==5.2.0

# Download sample testset
mkdir data && cd data
gdown https://drive.google.com/uc?id=1GcgddRm4GsjPKUOVdmWFzeF5gElCZfx2
unzip example_sequence.zip 
cd .. && mkdir results
```

### Run NavOCR!
```bash
# Remove visualize & save_results argument for fast inference
python run_inference.py   -c configs/ppyoloe/ppyoloe_crn_s_infer_only.yml   \
--infer_dir data/example_sequence/images  --visualize True  --save_results True
```

### Run NavOCR ROS node!
```bash
# Build and run as ROS 2 node (detection + OCR)
cd ~/ros2_ws
colcon build --packages-select navocr
source install/setup.bash

# If you encounter oneDNN compatibility issues on CPU, set these before running:
export FLAGS_enable_pir_api=0
export FLAGS_enable_pir_in_executor=0

ros2 run navocr navocr_with_ocr_node
```

## Training Model

Coming soon! (Dataset crawling, dataset preprocessing, model fine-tuning, ...)

## 🚧 Planned Updates
We're working on expanding support beyond store signboards detection model.
Stay tuned for upcoming features for broader navigation use cases.

- [x] Library migration due to a license issue (`ultralytics` -> `PaddleDetection`)
- [ ] Alternative inference for higher FPS (`PaddleDetection` is slow for video inference. (Currently 30 FPS with GPU))
- [ ] Model training scripts
- [x] Integration with text recognition (PaddleOCR)
- [ ] Room number and floor sign detection
- [ ] Directional guide text detection
- [x] Integration with SLAM packages via ROS (TextMap)

## License
This repository is licensed under the Apache License, Version 2.0.

This project includes code and configuration files derived from
PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection),
which is also licensed under the Apache License, Version 2.0.
